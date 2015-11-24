import Test.Hspec
import Test.QuickCheck
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuRAND as CuRAND

import HNN.NN.Mutable (createHandle)
import Test.HNN.NN.Mutable
import Test.HNN.Layers

main :: IO ()
main = do
  cudnn_handle <- createHandle
  cublas_handle <- Cublas.create
  hspec $ do
    -- HNN.NN.Mutable
    test_convolution2dFwd cudnn_handle
    test_pooling2dFwd cudnn_handle
    test_activationFwd cudnn_handle
    test_softmaxFwd cudnn_handle
    test_gemmFwd cublas_handle
    -- HNN.Layers
    test_dropout
    test_linear
    test_sumCols
    test_replicateAsCols
    test_convolution
    test_bias
    test_softmax
    test_log
    test_mlrCost
    test_pooling2d
    test_transformTensor
