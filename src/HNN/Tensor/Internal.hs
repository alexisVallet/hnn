{-# LANGUAGE GADTs, ForeignFunctionInterface, ScopedTypeVariables, TypeFamilies, DeriveGeneric, UndecidableInstances #-}
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
  , concatT
  , elementwiseMax
  , module Data.VectorSpace
  ) where
import Foreign
import Foreign.C
import Data.Proxy
import Control.Monad.Primitive
import System.IO.Unsafe
import Data.VectorSpace
import Data.AdditiveGroup
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SMV
import Data.Traversable
import GHC.Generics
import Data.Serialize
import Control.DeepSeq
import Data.Ratio

import qualified HNN.Tensor.Mutable.Internal as MT
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as CuBlas

data Tensor a = Tensor [Int] (ForeignPtr a)

data STensor a = STensor [Int] [a]
               deriving Generic

instance (Generic a, MT.TensorDataType a) => Generic (Tensor a) where
  type Rep (Tensor a) = Rep (STensor a)
  from t = from (STensor (shape t) (toList t))
  to rep = let STensor shape listData = to rep in
            fromList shape listData

instance (Serialize a, Generic a, MT.TensorDataType a) => Serialize (Tensor a)

instance (NFData a, Generic a, MT.TensorDataType a) => NFData (Tensor a)

instance (MT.TensorDataType a, Show a) => Show (Tensor a) where
  show t = "Tensor " ++ show (shape t) ++ " "  ++ show (take 10 $ toList t)

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

-- flattens and concatenates a list of vectors.
concatT :: (MT.TensorDataType a) => [Tensor a] -> Tensor a
concatT ts =
  fromList [sum $ fmap (product . shape) ts]
  $ concat $ fmap toList ts

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

instance (MT.TensorDataType a) => Fractional (Tensor a) where
  recip x = unsafePerformIO $ do
    res <- unsafeThaw x >>= MT.copy
    MT.inv res
    unsafeFreeze res
  fromRational r = fromInteger (numerator r) / fromInteger (denominator r)

elementwiseMax :: (MT.TensorDataType a) => Tensor a -> Tensor a -> Tensor a
elementwiseMax x y = unsafePerformIO $ do
  mx <- unsafeThaw x
  my <- unsafeThaw y >>= MT.copy
  MT.withDevicePtr mx $ \pmx -> do
    MT.withDevicePtr my $ \pmy -> do
      MT.rawMax pmx pmy (fromIntegral $ product $ shape x)
  unsafeFreeze my

fromRaw :: (MT.TensorDataType a) => (CUDA.DevicePtr a -> CSize -> IO ()) -> Tensor a -> Tensor a
fromRaw action x = unsafePerformIO $ do
  res <- unsafeThaw x >>= MT.copy
  let size = fromIntegral $ product $ shape x
  MT.withDevicePtr res $ \resptr -> do
    action resptr size
  unsafeFreeze res

instance (MT.TensorDataType a) => Floating (Tensor a) where
  pi = fromList [1] [pi]
  exp = fromRaw MT.rawExp
  log = fromRaw MT.rawLog
  sqrt = fromRaw MT.rawSqrt
  sin = fromRaw MT.rawSin
  cos = fromRaw MT.rawCos
  tan = fromRaw MT.rawTan
  asin = fromRaw MT.rawAsin
  acos = fromRaw MT.rawAcos
  atan = fromRaw MT.rawAtan
  sinh = fromRaw MT.rawSinh
  cosh = fromRaw MT.rawCosh
  tanh = fromRaw MT.rawTanh
  asinh = fromRaw MT.rawAsinh
  acosh = fromRaw MT.rawAcosh
  atanh = fromRaw MT.rawAtanh
  x**y = unsafePerformIO $ do
    mx <- unsafeThaw x
    my <- unsafeThaw y >>= MT.copy
    let size = fromIntegral $ product $ shape x
    MT.withDevicePtr mx $ \pmx -> do
      MT.withDevicePtr my $ \pmy -> do
        MT.rawPow pmx pmy size
    unsafeFreeze my

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

instance (MT.TensorDataType a) => InnerSpace (Tensor a) where
  t1 <.> t2 = foldr (+) 0 $ fmap (\(x1,x2) -> x1 * x2) $ zip (toList t1) (toList t2)
