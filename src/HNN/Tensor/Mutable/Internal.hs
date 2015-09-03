{-# LANGUAGE GADTs, ScopedTypeVariables #-}
module HNN.Tensor.Mutable.Internal (
  MTensor(..)
  , IOTensor
  , TensorDataType(..)
  , nbdims
  , dtype
  , shape
  , emptyTensor
  , withDevicePtr
  , fromList
  , toList
  , zeros
  , copy
  , threshInplace
  ) where
import Foreign
import Foreign.C
import Foreign.Marshal
import Foreign.Concurrent
import System.IO.Unsafe
import Data.List
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuRAND as CuRAND
import Data.Proxy
import System.IO.Error
import Control.Monad
import Control.Monad.Primitive
import Unsafe.Coerce

import qualified HNN.Internal.Cubits as Cubits

class (Cublas.Cublas a, Num a, Storable a) => TensorDataType a where
  datatype :: Proxy a -> CuDNN.DataType
  -- ad-hoc stuff to cleanly wrap low-level C APIs
  thresh :: CUDA.DevicePtr a -> CSize -> a -> CUDA.DevicePtr a -> IO ()
  rawMul :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawAdd :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawAbs :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSignum :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSubtract :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawNegate :: CUDA.DevicePtr a -> CSize -> IO ()
  rawScale :: a -> CUDA.DevicePtr a -> CSize -> IO ()
  -- curand stuff
  generateUniform :: CuRAND.Generator
                  -> CUDA.DevicePtr a
                  -> CSize
                  -> IO CuRAND.Status

instance TensorDataType CFloat where
  datatype = const CuDNN.float
  thresh = Cubits.thresh
  rawMul = Cubits.mul
  rawAdd = Cubits.add
  rawAbs = Cubits.tabs
  rawSignum = Cubits.tsignum
  rawSubtract = Cubits.subtract
  rawNegate = Cubits.tnegate
  rawScale = Cubits.scale
  generateUniform = CuRAND.generateUniform

instance TensorDataType CDouble where
  datatype = const CuDNN.double
  thresh = Cubits.threshDouble
  rawMul = Cubits.mulDouble
  rawAdd = Cubits.addDouble
  rawAbs = Cubits.tabsDouble
  rawSignum = Cubits.tsignumDouble
  rawSubtract = Cubits.subtractDouble
  rawNegate = Cubits.tnegateDouble
  rawScale = Cubits.scaleDouble
  generateUniform = CuRAND.generateUniformDouble

-- mutable tensor
data MTensor s a where
  MTensor :: (TensorDataType a)
          => [Int] -- shape
          -> ForeignPtr a -- data
          -> MTensor s a

type IOTensor = MTensor RealWorld

shape :: (PrimMonad m, TensorDataType a) => MTensor (PrimState m) a -> m [Int]
shape (MTensor shp _) = return shp

nbdims :: (PrimMonad m) => IOTensor a -> m Int
nbdims (MTensor shp _) = return $ length shp

dtype :: forall m a . (PrimMonad m, TensorDataType a)
      => MTensor (PrimState m) a
      -> m CuDNN.DataType
dtype _ = return $ datatype (Proxy :: Proxy a)

emptyTensor :: forall m a . (TensorDataType a, PrimMonad m)
            => [Int] -> m (MTensor (PrimState m) a)
emptyTensor shape = do
  res <- unsafePrimToPrim $ (emptyTensorIO shape :: IO (IOTensor a))
  return $ unsafeCoerce res

emptyTensorIO :: (TensorDataType a) => [Int] -> IO (IOTensor a)
emptyTensorIO shape = do
  let size = product shape
  dvcptr <- CUDA.mallocArray size
  let finalizer = CUDA.free dvcptr
  datafptr <- Foreign.Concurrent.newForeignPtr
              (CUDA.useDevicePtr dvcptr)
              finalizer
  return $ MTensor shape datafptr

withDevicePtr :: (Storable a) => IOTensor a -> (CUDA.DevicePtr a -> IO b) -> IO b
withDevicePtr (MTensor _ datafptr) action = do
  withForeignPtr datafptr $ \dataptr -> action (CUDA.DevicePtr dataptr)

makeTensor :: (TensorDataType a)
           => [Int]
           -> Ptr a
           -> IO (IOTensor a)
makeTensor shape dataptr = do
  let size = product shape
  tensor <- emptyTensor shape
  withDevicePtr tensor $ \dvcptr ->
    CUDA.pokeArray size dataptr dvcptr
  return tensor

fromList :: forall m a . (PrimMonad m, TensorDataType a)
         =>  [Int] -> [a] -> m (MTensor (PrimState m) a)
fromList shape content = do
  res <- unsafePrimToPrim $ (fromListIO shape content :: IO (IOTensor a))
  return $ unsafeCoerce res

fromListIO :: (TensorDataType a)
         => [Int]
         -> [a]
         -> IO (IOTensor a)
fromListIO shape datalist = do
  when (product shape /= length datalist) $ ioError $ userError
    "Shape is incompatible with provided data."
  withArray datalist $ \dataptr ->
    makeTensor shape dataptr

zeros :: forall m a . (PrimMonad m, TensorDataType a)
      => [Int] -> m (MTensor (PrimState m) a)
zeros shape = fromList shape $ take (product shape) $ repeat 0

toList :: (PrimMonad m, TensorDataType a)
       => MTensor (PrimState m) a -> m [a]
toList tensor = unsafePrimToPrim $ toListIO (unsafeCoerce tensor)

toListIO :: (TensorDataType a) => IOTensor a -> IO [a]
toListIO tensor = do
  tensorshape <- shape tensor
  withDevicePtr tensor (CUDA.peekListArray (product tensorshape))

copy :: forall m a . (PrimMonad m, TensorDataType a)
     => MTensor (PrimState m) a -> m (MTensor (PrimState m) a)
copy tensor = do
  res <- unsafePrimToPrim $ copyIO (unsafeCoerce tensor :: IOTensor a)
  return $ unsafeCoerce res

copyIO :: (TensorDataType a) => IOTensor a -> IO (IOTensor a)
copyIO tensor = do
  shp <- shape tensor
  out <- emptyTensorIO shp
  withDevicePtr tensor $ \tensorptr -> do
    withDevicePtr out $ \outptr -> do
      CUDA.copyArray (product shp) tensorptr outptr
      return out

threshInplace :: (PrimMonad m) => MTensor (PrimState m) CFloat -> CFloat -> m ()
threshInplace tensor threshold =
  unsafePrimToPrim $ threshInplaceIO (unsafeCoerce tensor :: IOTensor CFloat) threshold

threshInplaceIO :: IOTensor CFloat -> CFloat -> IO ()
threshInplaceIO tensor threshold = do
  size <- fmap (fromIntegral . product) $ shape tensor
  withDevicePtr tensor $ \tensorptr -> do
    thresh tensorptr size threshold tensorptr
