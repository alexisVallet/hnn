import Test.Hspec
import Test.QuickCheck
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuRAND as CuRAND

import HNN.NN.Mutable (createHandle)
import Test.HNN.Layers

main :: IO ()
main = do
  cudnn_handle <- createHandle
  cublas_handle <- Cublas.create
  hspec $ do
    -- HNN.Layers
    test_dropout
    test_linear
    test_sumCols
    test_replicateAsCols
    test_convolution
    test_softmax
    test_log
    test_mlrCost
    test_pooling2d
