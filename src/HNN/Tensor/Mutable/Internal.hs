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
  ) where
import Foreign
import Foreign.C
import Foreign.Marshal
import Foreign.Concurrent
import System.IO.Unsafe
import Data.List
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import Data.Proxy
import System.IO.Error
import Control.Monad
import Control.Monad.Primitive
import Unsafe.Coerce

class (Num a, Storable a) => TensorDataType a where
  datatype :: Proxy a -> CuDNN.DataType

instance TensorDataType Float where
  datatype = const CuDNN.float

instance TensorDataType Double where
  datatype = const CuDNN.double

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

toList :: (PrimMonad m, TensorDataType a)
       => MTensor (PrimState m) a -> m [a]
toList tensor = unsafePrimToPrim $ toListIO (unsafeCoerce tensor)

toListIO :: (TensorDataType a) => IOTensor a -> IO [a]
toListIO tensor = do
  tensorshape <- shape tensor
  withDevicePtr tensor (CUDA.peekListArray (product tensorshape))
