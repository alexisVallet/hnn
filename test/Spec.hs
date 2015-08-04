import Test.Hspec
import Test.QuickCheck

import HNN.NN.Mutable (createHandle)
import Test.HNN.NN.Mutable

main :: IO ()
main = do
  cudnn_handle <- createHandle
  hspec $ do
    test_convolution2dFwd cudnn_handle
    test_pooling2dFwd cudnn_handle
