{-# LANGUAGE GADTs, ForeignFunctionInterface, ScopedTypeVariables, TypeFamilies #-}
module HNN.Tensor.Internal (
  Tensor(..)
  , MT.TensorDataType
  , shape
  , nbdims
  , dtype
  , reshape
  , unsafeFreeze
  , unsafeThaw
  , fromList
  , toList
  , fromVector
  , toVector
  ) where
import Foreign
import Data.Proxy
import Control.Monad.Primitive
import System.IO.Unsafe
import Data.VectorSpace
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SMV

import qualified HNN.Tensor.Mutable.Internal as MT
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as CuBlas

data Tensor a where
  Tensor :: (MT.TensorDataType a)
         => [Int]
         -> ForeignPtr a
         -> Tensor a

shape :: Tensor a -> [Int]
shape (Tensor shp _) = shp

nbdims :: Tensor a -> Int
nbdims = length . shape

reshape :: [Int] -> Tensor a -> Tensor a
reshape newshp (Tensor oldshp ptr) =
  if product newshp == product oldshp
  then Tensor newshp ptr
  else error $ "Incompatible shapes for reshaping: " ++ show oldshp ++ ", " ++ show newshp

dtype :: forall a . (MT.TensorDataType a) => Tensor a -> CuDNN.DataType
dtype _ = MT.datatype (Proxy :: Proxy a)

-- mutable/unmutable conversions
unsafeFreeze :: (PrimMonad m) => MT.MTensor (PrimState m) a -> m (Tensor a)
unsafeFreeze (MT.MTensor shp ptr) = return $ Tensor shp ptr

unsafeThaw :: (PrimMonad m) => Tensor a -> m (MT.MTensor (PrimState m) a)
unsafeThaw (Tensor shp ptr) = return $ MT.MTensor shp ptr

-- conversion to/from lists
fromList :: (MT.TensorDataType a) => [Int] -> [a] -> Tensor a
fromList shape value = unsafePerformIO $ MT.fromList shape value >>= unsafeFreeze

toList :: (MT.TensorDataType a) => Tensor a -> [a]
toList t = unsafePerformIO $ unsafeThaw t >>= MT.toList

-- conversion to/from storable based vectors
fromVector :: (MT.TensorDataType a) => [Int] -> SV.Vector a -> Tensor a
fromVector shape value = unsafePerformIO $ do
  res <- MT.emptyTensor shape
  SV.unsafeWith value $ \vptr -> do
    MT.withDevicePtr res $ \resptr -> do
      CUDA.pokeArray (product shape) vptr resptr
  unsafeFreeze res

toVector :: (MT.TensorDataType a) => Tensor a -> SV.Vector a
toVector t = unsafePerformIO $ do
  let size = product $ shape t
  res <- SMV.new size
  mt <- unsafeThaw t
  SMV.unsafeWith res $ \resptr -> do
    MT.withDevicePtr mt $ \mtptr -> do
      CUDA.peekArray size mtptr resptr
  SV.unsafeFreeze res

-- elementwise Num instance
checkShape :: Tensor a -> Tensor a -> b -> b
checkShape t1 t2 x =
  if (shape t1 /= shape t2)
  then error $ "Incompatible shape for elementwise operation: " ++ show (shape t1) ++ ", " ++ show (shape t2)
  else x

-- broadcasting (only from scalars for now)
-- Very ugly CPU solution.
broadcast :: (MT.TensorDataType a) => Tensor a -> Tensor a -> (Tensor a, Tensor a)
broadcast t1 t2 =
  (broadcast_helper t1 t2, broadcast_helper t2 t1)
  where broadcast_helper t1' t2' =
          case shape t1' of
            [1] -> fromList (shape t2')
                   $ take (product $ shape t2')
                   $ repeat $ head $ toList t1'
            _ -> t1'

instance (MT.TensorDataType a) => Num (Tensor a) where
  _t1 + _t2 =
    let (t1, t2) = broadcast _t1 _t2 in
    checkShape t1 t2 $ unsafePerformIO $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy t2'
      let size = fromIntegral $ product $ shape t1
      MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> do
          MT.rawAdd t1ptr t3ptr size
      unsafeFreeze t3'
  _t1 * _t2 =
    let (t1, t2) = broadcast _t1 _t2 in
    checkShape t1 t2 $ unsafePerformIO $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy t2'
      let size = fromIntegral $ product $ shape t1
      MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> do
          MT.rawMul t1ptr t3ptr size
      unsafeFreeze t3'
  _t1 - _t2 =
    let (t1, t2) = broadcast _t1 _t2 in
    checkShape t1 t2 $ unsafePerformIO $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy t2'
      let size = fromIntegral $ product $ shape t1
      MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> do
          MT.rawSubtract t1ptr t3ptr size
      unsafeFreeze t3'
  negate t = unsafePerformIO $ do
    res <- unsafeThaw t >>= MT.copy
    let size = fromIntegral $ product $ shape t
    MT.withDevicePtr res $ \resptr -> do
      MT.rawNegate resptr size
    unsafeFreeze res
  signum t = unsafePerformIO $ do
    res <- unsafeThaw t >>= MT.copy
    let size = fromIntegral $ product $ shape t
    MT.withDevicePtr res $ \resptr -> do
      MT.rawSignum resptr size
    unsafeFreeze res
  fromInteger i = fromList [1] [fromIntegral i]

-- Vector space instance for Tensors.
instance (MT.TensorDataType a) => AdditiveGroup (Tensor a) where
  zeroV = fromInteger 0
  t1 ^+^ t2 = t1 + t2
  negateV t = negate t

instance (MT.TensorDataType a) => VectorSpace (Tensor a) where
  type Scalar (Tensor a) = a
  x *^ t = unsafePerformIO $ do -- TODO: somehow use Cublas's scal instead
    res <- unsafeThaw t >>= MT.copy
    let size = fromIntegral $ product $ shape t
    MT.withDevicePtr res $ \resptr -> do
      MT.rawScale x resptr size
    unsafeFreeze res
