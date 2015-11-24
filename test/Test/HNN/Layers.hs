{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, FlexibleContexts, TypeFamilies, DataKinds #-}
module Test.HNN.Layers where

import Prelude hiding (id, (.))
import Control.Category
import Foreign.C
import Test.Hspec
import Data.VectorSpace
import Control.Monad
import Control.Monad.Trans
import Data.HList.HList


import HNN.NN
import HNN.Layers
import HNN.Tensor
import HNN.Init

import Test.HNN.NN.NumericGrad

test_dropout :: Spec
test_dropout = describe "HNN.Layers.dropout" $ do
  let input = fromList [1,1,8,8] [1..8*8] :: Tensor CFloat
  it "does nothing for drop_proba = 0" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer $ dropout 0) (HLS HNil, input))
     `shouldReturn` toList input
  it "returns all zeros for drop_proba = 1" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer $ dropout 1) (HLS HNil, input))
     `shouldReturn` (take (8*8) $ repeat 0)

test_linear :: Spec
test_linear = describe "HNN.Layers.linear" $ do
  let
    layer = linear :: Layer GPU CFloat '[Tensor CFloat] (Tensor CFloat) (Tensor CFloat)
    x = fromList [2,2] [1,2,3,4] :: Tensor CFloat
    y = fromList [2,2] [5,6,7,8] :: Tensor CFloat
    expected = [19,22,43,50] :: [CFloat]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer layer) (HLS $ HCons y HNil,x))
     `shouldReturn` expected
  it "has analytic gradient close to numeric gradient" $ do
    let x = normal 0 0.1 [16,5]
        y = normal 0 0.1 [16,6]
        w = normal 0 0.1 [5,6] >>= \w' -> return $ HLS $ HCons w' HNil
    check_backward_layer layer w x y

test_sumCols :: Spec
test_sumCols = describe "HNN.Layers.sumCols" $ do
  let x = fromList [2,2] [1,2,3,4] :: Tensor CFloat
      expected = [4,6] :: [CFloat]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer sumCols) (HLS HNil,x))
     `shouldReturn` expected

test_sumRows :: Spec
test_sumRows = describe "HNN.Layers.sumRows" $ do
  let x = fromList [2,2] [1,2,3,4] :: Tensor CFloat
      expected = [3,7] :: [CFloat]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer sumRows) (HLS HNil,x))
     `shouldReturn` expected

allClose :: (TensorDataType a, Ord a) => Tensor a -> Tensor a -> Bool
allClose t1 t2 =
  all (\(x1,x2) -> abs (x1 - x2) / (abs x1 + abs x2) < 10E-2)
  $ zip (toList t1) (toList t2)

check_backward :: (TensorDataType a, Ord a, Show a,
                   Show inp, Show out, ToTensor inp,
                   ToTensor out, a ~ Scalar inp, a ~ Scalar out)
               => Diff GPU a inp out
               -> GPU inp
               -> GPU out
               -> Expectation
check_backward layer input upgrad = runGPU 42 $ do
  x <- input
  y <- upgrad
  num_x' <- genericNumericBwd (\x1 -> forward layer x1) x y
  analytic_x' <- backward layer x y
  let (t_num_x', _) = toTensor num_x'
      (t_analytic_x', _) = toTensor analytic_x'
  when (not $ allClose t_num_x' t_analytic_x') $ do
    liftIO $ expectationFailure $ "Numeric and analytic gradient do not match:\nNumeric: " ++ show num_x' ++ "\nAnalytic: " ++ show analytic_x'

check_backward_layer layer w x upgrad = check_backward (unLayer layer) inp upgrad
  where inp = do
          w' <- w
          x' <- x
          return (w',x')


test_convolution :: Spec
test_convolution = describe "HNN.Layers.convolution2d" $ do
  let conv = convolution2d convolution_fwd_algo_implicit_gemm
             (1,1) (1,1) (1,1)
  it "has a working forward pass" $ do
    let filter = fromList [1,1,3,3] [0, 1, 0,
                                     1, 0, 1,
                                     0, 1, 0] :: Tensor CFloat
        img = fromList [1,1,3,3] [1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9] :: Tensor CFloat
        expected_out = fromList [1,1,3,3] [6,  9,  8,
                                           13, 20, 17,
                                           12, 21, 14] :: Tensor CFloat
    (fmap toList $ runGPU 42 $ forward (unLayer conv) (HLS $ HCons filter HNil,img)) `shouldReturn` toList expected_out
  it "has a correct backward pass" $ do
    let x = normal 0 0.1 [1,1,8,8]
        y = normal 0 0.1 [1,1,8,8]
        w = fmap (\w -> HLS $ HCons w HNil) $ normal 0 0.1 [1,1,3,3]
    check_backward_layer conv w x y

test_bias :: Spec
test_bias = describe "HNN.Layers.bias" $ do
  it "returns the right result for a simple example" $ do
    let src = fromList [1,3,2,2] [1,2,3,4,5,6,
                                  7,8,9,10,11,12] :: Tensor CFloat
        w = fromList [1,3,1,1] [1, 2, 3] :: Tensor CFloat
        expected_out = [2,3,4,5,7,8,9,10,12,13,14,15] :: [CFloat]
    (fmap toList $ runGPU 42 $ forward (unLayer bias) (HLS $ HCons w HNil,src))
      `shouldReturn` expected_out
  it "has analytic gradient close to numerical gradient" $ do
    let src = normal 0 0.1 [1,5,2,2] :: GPU (Tensor CFloat)
        w = normal 0 0.1 [1,5,1,1] :: GPU (Tensor CFloat)
        upgrad = normal 0 0.1 [1,5,2,2] :: GPU (Tensor CFloat)
    check_backward_layer bias (fmap (HLS . flip HCons HNil) w) src upgrad

splitEvery :: Int -> [a] -> [[a]]
splitEvery n [] = []
splitEvery n xs = let (start,rest) = splitAt n xs in
  [start] ++ splitEvery n rest

naive_softmax :: Tensor CFloat -> Tensor CFloat
naive_softmax t = let [n,m] = shape t in
  fromList [n,m] $ concat $ fmap softmaxLine $ splitEvery m $ toList t
  where softmaxLine xs = let denom = sum $ fmap exp xs in
          fmap (\x -> exp x / denom) xs

naive_mlrCost :: Tensor CFloat -> Tensor CFloat -> CFloat
naive_mlrCost l x =
  let softmaxed = naive_softmax x
      [n,m] = shape x
      labels = toList l
  in
   (-1/fromIntegral n)
   * (sum
      $ fmap log
      $ fmap sum
      $ splitEvery m
      $ zipWith (*) labels (toList softmaxed)                )

test_softmax :: Spec
test_softmax = describe "softmax" $ do
  let layer = softmax 8 8 :: Layer GPU CFloat '[] (Tensor CFloat) (Tensor CFloat)
  it "CPU and GPU naive softmaxes return the same results" $ runGPU 42 $ do
    x <- normal 0 0.1 [8,8]
    let expected = naive_softmax x
    actual <- forward (unLayer layer) (HLS HNil,x)
    when (not $ allClose expected actual) $ do
      liftIO $ expectationFailure $
        "Naive and GPU algorithm return different results:\nnaive: "
        ++ show expected ++ "\ncudnn: " ++ show actual
  it "has a numerically correct backward pass" $ do
    check_backward_layer layer (return $ HLS HNil) (normal 0 0.1 [8,8]) (normal 0 0.1 [8,8])

test_replicateAsCols :: Spec
test_replicateAsCols = describe "HNN.Layers.replicateAsCols" $ do
  let x = fromList [4] [1,2,3,4] :: Tensor CFloat
      n = 5
      expected = [1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer $ replicateAsCols n) (HLS HNil, x)) `shouldReturn` expected
  it "has a backward pass close to the numerical backward pass" $ do
    check_backward_layer (replicateAsCols n :: Layer GPU CFloat '[] (Tensor CFloat) (Tensor CFloat)) (return $ HLS $ HNil) (normal 0 0.1 [56]) (normal 0 0.1 [56,n])

test_log :: Spec
test_log = describe "HNN.Layers.log" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    let x = uniform [8,8] >>= return . (fromList [1] [10E-5] +)
        y = uniform [8,8] >>= return . (fromList [1] [10E-5] +)
    check_backward_layer (llog :: Layer GPU CFloat '[] (Tensor CFloat) (Tensor CFloat)) (return $ HLS HNil) x y

test_mlrCost :: Spec
test_mlrCost = describe "HNN.Layers.mlrCost" $ do
  let labels = fromList [8,8] [1,0,0,0,1,1,1,0,
                               0,1,1,1,1,1,0,1,
                               1,1,1,1,1,1,0,0,
                               0,1,0,0,0,0,1,0,
                               1,0,0,0,1,1,1,0,
                               0,1,1,1,1,1,0,1,
                               1,1,1,1,1,1,0,0,
                               0,1,0,0,0,0,1,0] :: Tensor (CFloat)
      x = fromList [8,8] [1..8*8] :: Tensor (CFloat)
      expected = naive_mlrCost labels x
  it "returns the same results as a naive CPU implementation" $ do
    (runGPU 42 $ forward (unLayer $ mlrCost 8 8) (HLS HNil,(labels,x))) `shouldReturn` expected
  it "has an analytic gradient close to the numeric gradient" $ do
    check_backward_layer (mlrCost 8 8) (return $ HLS HNil) (return (labels,x)) (return 1)

test_pooling2d :: Spec
test_pooling2d = describe "HNN.Layers.pooling2d" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    check_backward_layer (pooling2d pooling_max (2,2) (0,0) (2,2)) (return $ HLS HNil) (normal 0 0.1 [1,1,4,4]) (normal 0 0.1 [1,1,2,2] :: GPU (Tensor CFloat))

test_transformTensor :: Spec
test_transformTensor = describe "HNN.Layers.transformTensor" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    check_backward_layer (transformTensor nhwc nchw) (return $ HLS HNil) (normal 0 0.1 [1,3,2,4] :: GPU (Tensor CFloat)) (normal 0 0.1 [1,4,3,2] :: GPU (Tensor CFloat))
