module Test.HNN.NN.Mutable where

import Foreign.C
import Control.Monad.ST
import Test.Hspec

import qualified Foreign.CUDA.CuDNN as CuDNN

import HNN.NN.Mutable
import HNN.Tensor.Mutable

test_convolution2dFwd :: CuDNN.Handle -> Spec
test_convolution2dFwd handle = describe "HNN.NN.Mutable.convolution2dFwd" $ do
  let filter = [0, 1, 0,
                1, 0, 1,
                0, 1, 0] :: [CFloat]
      fmap = [1, 2, 3,
              4, 5, 6,
              7, 8, 9] :: [CFloat]
      expected_out = [6,  9,  8,
                      13, 20, 17,
                      12, 21, 14] :: [CFloat]
      convresult = runST $ do
        filter_tensor <- fromList [1,1,3,3] filter
        fmap_tensor <- fromList [1,1,3,3] fmap
        conv_tensor <- convolution2dFwd handle
                       CuDNN.convolution_fwd_algo_implicit_gemm
                       (1,1) (1,1) (1,1) fmap_tensor filter_tensor
        toList conv_tensor
  it "Returns the right result for a simple example" $ do
    convresult `shouldBe` expected_out

test_pooling2dFwd :: CuDNN.Handle -> Spec
test_pooling2dFwd handle = describe "HNN.NN.Mutable.pooling2dFwd" $ do
  let fmap = [1..16] :: [CFloat]
      expected_out = [6, 8,
                      14,16]
      (mpresult,mpshape) = runST $ do
        fmap_tensor <- fromList [1,1,4,4] fmap
        mp_tensor <- pooling2dFwd handle CuDNN.pooling_max
                     (3,3) (1,1) (2,2) fmap_tensor
        coeffs <- toList mp_tensor
        res_shape <- shape mp_tensor
        return (coeffs,res_shape)
  it "Returns the right result for a simple example" $ do
    mpresult `shouldBe` expected_out
  it "Returns a result which has the right shape" $ do
    mpshape `shouldBe` [1,1,2,2]
