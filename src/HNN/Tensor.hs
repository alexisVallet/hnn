{-# LANGUAGE GADTs, ForeignFunctionInterface, ScopedTypeVariables #-}
module HNN.Tensor (
  MTensor
  , IOTensor
  , TensorDataType(..)
  , nbdims
  , dtype
  , shape
  , emptyTensor
  , makeTensor
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
import Control.Monad.ST

class (Num a) => TensorDataType a where
  datatype :: Proxy a -> CuDNN.DataType

instance TensorDataType Float where
  datatype = const CuDNN.float

instance TensorDataType Double where
  datatype = const CuDNN.double

-- mutable tensor
data MTensor s a where
  MTensor :: (TensorDataType a, Storable a)
          => [Int] -- shape
          -> ForeignPtr a -- data
          -> MTensor s a

type IOTensor = MTensor RealWorld

shape :: MTensor s a -> [Int]
shape (MTensor shp _) = shp

nbdims :: MTensor s a -> Int
nbdims (MTensor shp _) = length shp

dtype :: forall s a . (TensorDataType a) => MTensor s a -> CuDNN.DataType
dtype _ = datatype (Proxy :: Proxy a)

emptyTensor :: (TensorDataType a, Storable a) => [Int] -> IO (IOTensor a)
emptyTensor shape = do
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

makeTensor :: (TensorDataType a, Storable a)
           => [Int]
           -> Ptr a
           -> IO (IOTensor a)
makeTensor shape dataptr = do
  let size = product shape
  tensor <- emptyTensor shape
  withDevicePtr tensor $ \dvcptr ->
    CUDA.pokeArray size dataptr dvcptr
  return tensor

fromList :: (TensorDataType a, Storable a)
         => [Int]
         -> [a]
         -> IO (IOTensor a)
fromList shape datalist = do
  when (product shape /= length datalist) $ ioError $ userError
    "Shape is incompatible with provided data."
  withArray datalist $ \dataptr ->
    makeTensor shape dataptr

toList :: (Storable a) => IOTensor a -> IO [a]
toList tensor = withDevicePtr tensor (CUDA.peekListArray (product $ shape tensor))
