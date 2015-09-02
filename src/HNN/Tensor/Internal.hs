{-# LANGUAGE GADTs, ForeignFunctionInterface, ScopedTypeVariables #-}
module HNN.Tensor.Internal (
  Tensor(..)
  , MT.TensorDataType
  , shape
  , nbdims
  , dtype
  , unsafeFreeze
  , unsafeThaw
  , fromList
  , toList
  ) where
import Foreign
import Data.Proxy
import Control.Monad.Primitive
import System.IO.Unsafe

import qualified HNN.Tensor.Mutable.Internal as MT
import qualified Foreign.CUDA.CuDNN as CuDNN

data Tensor a where
  Tensor :: (MT.TensorDataType a)
         => [Int]
         -> ForeignPtr a
         -> Tensor a

shape :: Tensor a -> [Int]
shape (Tensor shp _) = shp

nbdims :: Tensor a -> Int
nbdims = length . shape

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

-- elementwise Num instance
checkShape :: Tensor a -> Tensor a -> b -> b
checkShape t1 t2 x =
  if shape t1 /= shape t2
  then error $ "Incompatible shape for elementwise operation: " ++ show (shape t1) ++ ", " ++ show (shape t2)
  else x

instance (MT.TensorDataType a) => Num (Tensor a) where
  t1 + t2 = checkShape t1 t2 $ unsafePerformIO $ do
    t1' <- unsafeThaw t1
    t2' <- unsafeThaw t2
    t3' <- MT.copy t2'
    let size = fromIntegral $ product $ shape t1
    MT.withDevicePtr t1' $ \t1ptr -> do
      MT.withDevicePtr t3' $ \t3ptr -> do
        MT.rawAdd t1ptr t3ptr size
    unsafeFreeze t3'
  t1 * t2 = checkShape t1 t2 $ unsafePerformIO $ do
    t1' <- unsafeThaw t1
    t2' <- unsafeThaw t2
    t3' <- MT.copy t2'
    let size = fromIntegral $ product $ shape t1
    MT.withDevicePtr t1' $ \t1ptr -> do
      MT.withDevicePtr t3' $ \t3ptr -> do
        MT.rawMul t1ptr t3ptr size
    unsafeFreeze t3'
  t1 - t2 = checkShape t1 t2 $ unsafePerformIO $ do
    t1' <- unsafeThaw t1
    t2' <- unsafeThaw t2
    t3' <- MT.copy t2'
    let size = fromIntegral $ product $ shape t1
    MT.withDevicePtr t1' $ \t1ptr -> do
      MT.withDevicePtr t3' $ \t3ptr -> do
        MT.rawSubtract t1ptr t3ptr size
    unsafeFreeze t3'
  signum t = unsafePerformIO $ do
    res <- unsafeThaw t >>= MT.copy
    let size = fromIntegral $ product $ shape t
    MT.withDevicePtr res $ \resptr -> do
      MT.rawSignum resptr size
    unsafeFreeze res
  fromInteger i = fromList [1] [fromIntegral i]
