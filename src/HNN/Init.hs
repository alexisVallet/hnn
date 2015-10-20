module HNN.Init where

import HNN.Layers.Internal
import Control.Lens
import qualified Foreign.CUDA.CuRAND as CuRAND
import Control.Monad.Trans
import HNN.Tensor
import qualified HNN.Tensor.Mutable.Internal as MT

import HNN.NN
import HNN.Layers

-- generating random tensors
uniform :: (TensorDataType a) => [Int] -> GPU (Tensor a)
uniform shape = do
  gen <- view generator
  liftIO $ do
    res <- MT.emptyTensor shape
    MT.withDevicePtr res $ \resptr -> do
      MT.generateUniform gen resptr (fromIntegral $ product shape)
    unsafeFreeze res

normal :: (TensorDataType a) => a -> a -> [Int] -> GPU (Tensor a)
normal mean std shape = do
  gen <- view generator
  liftIO $ do
    res <- MT.emptyTensor shape
    MT.withDevicePtr res $ \resptr -> do
      MT.generateNormal gen resptr (fromIntegral $ product shape) mean std
    unsafeFreeze res

logNormal :: (TensorDataType a) => a -> a -> [Int] -> GPU (Tensor a)
logNormal mean std shape = do
  gen <- view generator
  liftIO $ do
    res <- MT.emptyTensor shape
    MT.withDevicePtr res $ \resptr -> do
      MT.generateLogNormal gen resptr (fromIntegral $ product shape) mean std
    unsafeFreeze res

