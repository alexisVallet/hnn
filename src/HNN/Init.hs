module HNN.Init where

import HNN.Layers.Internal
import Control.Lens
import qualified Foreign.CUDA.CuRAND as CuRAND
import Control.Monad.Trans
import HNN.Tensor
import qualified HNN.Tensor.Mutable.Internal as MT
import Data.Proxy

import HNN.NN
import HNN.Layers

-- generating random tensors
uniform :: forall a s . (TensorDataType a, Shape s) => GPU (Tensor s a)
uniform = do
  gen <- view generator
  liftIO $ do
    res <- MT.emptyTensor
    MT.withDevicePtr res $ \resptr -> do
      MT.generateUniform gen resptr $ fromIntegral $ size (Proxy :: Proxy s)
    unsafeFreeze res

normal :: forall a s . (TensorDataType a, Shape s) => a -> a -> GPU (Tensor s a)
normal mean std = do
  gen <- view generator
  liftIO $ do
    res <- MT.emptyTensor
    MT.withDevicePtr res $ \resptr -> do
      MT.generateNormal gen resptr (fromIntegral $ size (Proxy :: Proxy s)) mean std
    unsafeFreeze res

logNormal :: forall a s . (TensorDataType a, Shape s) => a -> a -> GPU (Tensor s a)
logNormal mean std = do
  gen <- view generator
  liftIO $ do
    res <- MT.emptyTensor
    MT.withDevicePtr res $ \resptr -> do
      MT.generateLogNormal gen resptr (fromIntegral $ size (Proxy :: Proxy s)) mean std
    unsafeFreeze res

