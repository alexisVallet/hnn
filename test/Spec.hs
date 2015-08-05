import Test.Hspec
import Test.QuickCheck
import qualified Foreign.CUDA.Cublas as Cublas

import HNN.NN.Mutable (createHandle)
import Test.HNN.NN.Mutable

main :: IO ()
main = do
  cudnn_handle <- createHandle
  cublas_handle <- Cublas.create
  hspec $ do
    test_convolution2dFwd cudnn_handle
    test_pooling2dFwd cudnn_handle
    test_activationFwd cudnn_handle
    test_softmaxFwd cudnn_handle
    test_gemmFwd cublas_handle
