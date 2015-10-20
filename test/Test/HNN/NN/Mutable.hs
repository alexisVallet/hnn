module Test.HNN.NN.Mutable where

import Foreign.C
import Control.Monad
import Control.Monad.ST
import Test.Hspec
import Data.List
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuRAND as CuRAND

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
  it "returns the right result for a simple example" $ do
    convresult `shouldBe` expected_out

test_pooling2dFwd :: CuDNN.Handle -> Spec
test_pooling2dFwd handle = describe "HNN.NN.Mutable.pooling2dFwd" $ do
  let fmap = [1..16] :: [CFloat]
      expected_out = [6, 8,
                      14,16]
      (mpresult,mpshape) = runST $ do
        fmap_tensor <- fromList [1,1,4,4] fmap
        mp_tensor <- pooling2dFwd handle CuDNN.pooling_max
                     (2,2) (0,0) (2,2) fmap_tensor
        coeffs <- toList mp_tensor
        res_shape <- shape mp_tensor
        return (coeffs,res_shape)
  it "returns the right result for a simple example" $ do
    mpresult `shouldBe` expected_out
  it "returns a result which has the right shape" $ do
    mpshape `shouldBe` [1,1,2,2]

test_activationFwd :: CuDNN.Handle -> Spec
test_activationFwd handle = describe "HNN.NN.Mutable.activationFwd" $ do
  let fmap = [-2,  0,   1,
              -8, -15,  12,
              14,  72, -0.5] :: [CFloat]
      expected_out = [0, 0, 1,
                      0, 0, 12,
                      14, 72, 0] :: [CFloat]
      actual_out = runST $ do
        fmap_tensor <- fromList [1,1,3,3] fmap
        act_tensor <- activationFwd handle CuDNN.activation_relu
                      fmap_tensor
        toList act_tensor
  it "returns the right result for a simple example" $ do
    actual_out `shouldBe` expected_out

argsort :: (Ord a) => [a] -> [Int]
argsort = map fst . sortBy (\(_,x) (_,y) -> compare x y) . zip [0..]

test_softmaxFwd :: CuDNN.Handle -> Spec
test_softmaxFwd handle = describe "HNN.NN.Mutable.softmaxFwd" $ do
  let fmaps = [-2,  0,   1,
              -8, -15,  12,
              14,  72, -0.5] :: [CFloat]
      softmax_res = runST $ do
        fmap_tensor <- fromList [3,3,1,1] fmaps
        softmax_tensor <- softmaxFwd handle CuDNN.softmax_accurate
                          CuDNN.softmax_mode_instance fmap_tensor
        toList softmax_tensor
      tolines [x11,x12,x13,x21,x22,x23,x31,x32,x33] =
        [[x11,x12,x13],
         [x21,x22,x23],
         [x31,x32,x33]]
      eps = 10E-5
  it "preserves ordering on a simple example" $ do
    forM_ (zip (tolines fmaps) (tolines softmax_res)) $ \(fl,sl) -> do
      argsort fl `shouldBe` argsort sl
  it "outputs between 0 and 1" $ do
    softmax_res `shouldSatisfy` all (\x -> x >= 0 && x <= 1)
  it "has rows summing to 1" $ do
    tolines softmax_res `shouldSatisfy` all ((\x -> x >= 1-eps && x <= 1+eps) . sum)

test_gemmFwd :: Cublas.Handle -> Spec
test_gemmFwd handle = describe "HNN.NN.Mutable.gemmFwd" $ do
  let a = [1, 2, 3,
           4, 5, 6,
           7, 8, 9] :: [CFloat]
      b = [1, 2,
           3, 4,
           5, 6] :: [CFloat]
      c = [1, 2,
           3, 5,
           7, 11] :: [CFloat]
      alpha = 1/2 :: CFloat
      beta = 1/3 :: CFloat
      expected_out = [11.33333, 14.66666,
                      25.5,     33.66666,
                      40.33333, 53.66666]
      actual_out = runST $ do
        a_tensor <- fromList [3,3] a
        b_tensor <- fromList [3,2] b
        c_tensor <- fromList [3,2] c
        gemmFwd handle Cublas.N Cublas.N alpha a_tensor b_tensor
          beta c_tensor
        toList c_tensor
      eps=10E-4
  it "returns the right result for a simple example" $ do
    actual_out `shouldSatisfy` (
      all (\(x,y) -> abs (x-y) <= eps)
      . zip expected_out
      )
