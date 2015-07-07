{-# LANGUAGE GADTs, ForeignFunctionInterface #-}
module HNN.Tensor (
  Tensor
  , TensorDataType
  , nbdims
  , dtype
  , shape
  , stride
  , make4dTensor
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

class TensorDataType a where
  toCudnnType :: Proxy a -> CuDNN.DataType

instance TensorDataType CFloat where
  toCudnnType = const CuDNN.float

instance TensorDataType CDouble where
  toCudnnType = const CuDNN.double

data Tensor a where
  Tensor :: (TensorDataType a)
         => CuDNN.TensorDescriptor
         -> Int -- nbdims
         -> ForeignPtr a -- data
         -> Tensor a

-- Basic accessors for shape, stride, datatype.
desc :: Tensor a -> CuDNN.TensorDescriptor
desc (Tensor desc _ _) = desc

nbdims :: Tensor a -> Int
nbdims (Tensor _ nbdims _) = nbdims

descriptor :: Tensor a -> (CuDNN.DataType, Int, [Int], [Int])
descriptor t = unsafePerformIO $ do
  alloca $ \dtypeptr ->
    alloca $ \nbdimsptr ->
      alloca $ \dimsptr ->
        alloca $ \stridesptr -> do
          CuDNN.getTensorNdDescriptor
            (desc t) (fromIntegral $ nbdims t) dtypeptr nbdimsptr dimsptr stridesptr
          dtype <- peek dtypeptr
          dims <- peekArray (fromIntegral $ nbdims t) dimsptr
          strides <- peekArray (fromIntegral $ nbdims t) stridesptr
          return (dtype, (nbdims t), fmap fromIntegral dims, fmap fromIntegral strides)

dtype :: Tensor a -> CuDNN.DataType
dtype t = dtp where (dtp, _, _, _) = descriptor t

shape :: Tensor a -> [Int]
shape t = shp where (_, _, shp, _) = descriptor t

stride :: Tensor a -> [Int]
stride t = str where (_, _, _, str) = descriptor t


-- Creating tensors from raw memory.
-- 4d tensors.
make4dTensor :: (Storable a, TensorDataType a)
             => Proxy a
             -> (Int, Int, Int, Int)
             -> Ptr a
             -> IO (Tensor a)
make4dTensor proxy (n,c,h,w) dataptr = do
  -- create the tensor descriptor.
  desc <- alloca $ \descptr -> do
    status <- CuDNN.createTensorDescriptor descptr
    if status == CuDNN.success
      then peek descptr
      else do
        errMsg <- CuDNN.getErrorString status >>= peekCString
        ioError $ userError $ "Couldn't create tensor descriptor: " ++ errMsg
  -- set it
  let [cn,cc,ch,cw] = map fromIntegral [n,c,h,w]
  status <- CuDNN.setTensor4dDescriptor desc CuDNN.nchw (toCudnnType proxy) cn cc ch cw
  when (status /= CuDNN.success) $ do
    errMsg <- CuDNN.getErrorString status >>= peekCString
    ioError $ userError $ "Couldn't set tensor descriptor: " ++ errMsg
  -- allocate and fill the data in device memory
  let size = n*c*h*w
  dvcptr <- CUDA.mallocArray size
  CUDA.pokeArray size dataptr dvcptr
  let finalizer = CUDA.free dvcptr
  datafptr <- Foreign.Concurrent.newForeignPtr (CUDA.useDevicePtr dvcptr) finalizer
  return $ Tensor desc 4 datafptr
