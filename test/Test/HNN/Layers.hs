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
import GHC.TypeLits
import Data.Proxy

import HNN.NN
import HNN.Layers
import HNN.Tensor
import HNN.Init

import Test.HNN.NumericGrad

test_dropout :: Spec
test_dropout = describe "HNN.Layers.dropout" $ do
  let input = fromList [1..8*8] :: Tensor [1,1,8,8] CFloat
  it "does nothing for drop_proba = 0" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer $ dropout 0) (HLS HNil, input))
     `shouldReturn` toList input
  it "returns all zeros for drop_proba = 1" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer $ dropout 1) (HLS HNil, input))
     `shouldReturn` (take (8*8) $ repeat 0)

test_linear :: Spec
test_linear = describe "HNN.Layers.linear" $ do
  it "returns the right result for a simple example" $ do
    let
      x = fromList [1,2,3,4] :: Tensor [2,2] CFloat
      y = fromList [5,6,7,8] :: Tensor [2,2] CFloat
      expected = [19,22,43,50] :: [CFloat]
    (fmap toList $ runGPU 42 $ forward (unLayer linear) (HLS $ HCons y HNil,x))
     `shouldReturn` expected
  it "has analytic gradient close to numeric gradient" $ do
    let x = normal 0 0.1 :: GPU (Tensor [16,5] CFloat)
        y = normal 0 0.1 :: GPU (Tensor [16,6] CFloat)
        w = do
          w' <- normal 0 0.1 :: GPU (Tensor [5,6] CFloat)
          return $ HLS $ HCons w' HNil
    check_backward_layer linear w x y

test_sumCols :: Spec
test_sumCols = describe "HNN.Layers.sumCols" $ do
  let x = fromList [1,2,3,4] :: Tensor [2,2] CFloat
      expected = [4,6] :: [CFloat]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer sumCols) (HLS HNil,x))
     `shouldReturn` expected

test_sumRows :: Spec
test_sumRows = describe "HNN.Layers.sumRows" $ do
  let x = fromList [1,2,3,4] :: Tensor [2,2] CFloat
      expected = [3,7] :: [CFloat]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer sumRows) (HLS HNil,x))
     `shouldReturn` expected

allClose :: (TensorDataType a, Ord a, Shape s) => Tensor s a -> Tensor s a -> Bool
allClose t1 t2 =
  all (\(x1,x2) -> abs (x1 - x2) / (abs x1 + abs x2) < 10E-2)
  $ zip (toList t1) (toList t2)

check_backward :: (TensorDataType a, Ord a, Show a,
                   Show inp, Show out, ToTensor inp,
                   ToTensor out, KnownNat (ConcatSize inp),
                   KnownNat (ConcatSize out), a ~ Scalar inp, a ~ Scalar out)
               => Diff GPU a inp out
               -> GPU inp
               -> GPU out
               -> Expectation
check_backward layer input upgrad = runGPU 42 $ do
  x <- input
  y <- upgrad
  num_x' <- genericNumericBwd (\x1 -> forward layer x1) x y
  analytic_x' <- backward layer x y
  let t_num_x' = toTensor num_x'
      t_analytic_x' = toTensor analytic_x'
  when (not $ allClose t_num_x' t_analytic_x') $ do
    liftIO $ expectationFailure $ "Numeric and analytic gradient do not match:\nNumeric: " ++ show num_x' ++ "\nAnalytic: " ++ show analytic_x'

check_backward_layer layer w x upgrad = check_backward (unLayer layer) inp upgrad
  where inp = do
          w' <- w
          x' <- x
          return (w',x')


test_convolution :: Spec
test_convolution = describe "HNN.Layers.convolution2d" $ do
  it "has a working forward pass" $ do
    let conv = convolution2d
               (Proxy :: Proxy [[1,1],[1,1]])
               convolution_fwd_algo_implicit_gemm
        filter = fromList [0, 1, 0,
                           1, 0, 1,
                           0, 1, 0] :: Tensor [1,1,3,3] CFloat
        img = fromList [1, 2, 3,
                        4, 5, 6,
                        7, 8, 9] :: Tensor [1,1,3,3] CFloat
        expected_out = fromList [6,  9,  8,
                                 13, 20, 17,
                                 12, 21, 14] :: Tensor [1,1,3,3] CFloat
    (fmap toList $ runGPU 42 $ forward (unLayer conv) (HLS $ HCons filter HNil,img)) `shouldReturn` toList expected_out
  it "has a correct backward pass" $ do
    let conv = convolution2d
               (Proxy :: Proxy [[1,1],[1,1]])
               convolution_fwd_algo_implicit_gemm
        x = normal 0 0.1 :: GPU (Tensor [1,1,8,8] CFloat)
        y = normal 0 0.1 :: GPU (Tensor [1,1,8,8] CFloat)
        w = do
          w' <- normal 0 0.1 :: GPU (Tensor [1,1,3,3] CFloat)
          return $ HLS $ w' `HCons` HNil
    check_backward_layer conv w x y

splitEvery :: Int -> [a] -> [[a]]
splitEvery n [] = []
splitEvery n xs = let (start,rest) = splitAt n xs in
  [start] ++ splitEvery n rest

naive_softmax :: forall m n
              . (KnownNat n, KnownNat m) => Tensor [n,m] CFloat -> Tensor [n,m] CFloat
naive_softmax t = fromList $ concat $ fmap softmaxLine $ splitEvery m $ toList t
  where
    [n,m] = shape (Proxy :: Proxy [n,m])
    softmaxLine xs = let denom = sum $ fmap exp xs in
                      fmap (\x -> exp x / denom) xs

naive_mlrCost :: forall m n . (KnownNat n, KnownNat m)
              => Tensor [n,m] CFloat -> Tensor [n,m] CFloat -> CFloat
naive_mlrCost l x =
  let softmaxed = naive_softmax x
      [n,m] = shape (Proxy :: Proxy [n,m])
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
  it "CPU and GPU naive softmaxes return the same results" $ runGPU 42 $ do
    x <- normal 0 0.1 :: GPU (Tensor [8,8] CFloat)
    let expected = naive_softmax x
    actual <- forward (unLayer softmax) (HLS HNil,x)
    when (not $ allClose expected actual) $ do
      liftIO $ expectationFailure $
        "Naive and GPU algorithm return different results:\nnaive: "
        ++ show expected ++ "\ncudnn: " ++ show actual
  it "has a numerically correct backward pass" $ do
    check_backward_layer softmax (return $ HLS HNil)
      (normal 0 0.1 :: GPU (Tensor [8,8] CFloat))
      (normal 0 0.1 :: GPU (Tensor [8,8] CFloat))

test_replicateAsCols :: Spec
test_replicateAsCols = describe "HNN.Layers.replicateAsCols" $ do
  let x = fromList [1,2,3,4] :: Tensor '[4] CFloat
      expected = [1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runGPU 42 $ forward (unLayer $ replicateAsCols (Proxy :: Proxy 5)) (HLS HNil, x)) `shouldReturn` expected
  it "has a backward pass close to the numerical backward pass" $ do
    check_backward_layer (replicateAsCols (Proxy :: Proxy 5)) (return $ HLS $ HNil)
      (normal 0 0.1 :: GPU (Tensor '[56] CFloat))
      (normal 0 0.1 :: GPU (Tensor [56,5] CFloat))

test_log :: Spec
test_log = describe "HNN.Layers.log" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    let x = uniform >>= return . (10E-5 +) :: GPU (Tensor [8,8] CFloat)
        y = uniform >>= return . (10E-5 +) :: GPU (Tensor [8,8] CFloat)
    check_backward_layer llog (return $ HLS HNil) x y

test_mlrCost :: Spec
test_mlrCost = describe "HNN.Layers.mlrCost" $ do
  let labels = fromList [1,0,0,0,1,1,1,0,
                         0,1,1,1,1,1,0,1,
                         1,1,1,1,1,1,0,0,
                         0,1,0,0,0,0,1,0,
                         1,0,0,0,1,1,1,0,
                         0,1,1,1,1,1,0,1,
                         1,1,1,1,1,1,0,0,
                         0,1,0,0,0,0,1,0] :: Tensor [8,8] (CFloat)
      x = fromList [1..8*8] :: Tensor [8,8] (CFloat)
      expected = naive_mlrCost labels x
  it "returns the same results as a naive CPU implementation" $ do
    (runGPU 42 $ forward (unLayer $ mlrCost) (HLS HNil,(labels,x)))
      `shouldReturn` expected
  it "has an analytic gradient close to the numeric gradient" $ do
    check_backward_layer mlrCost (return $ HLS HNil) (return (labels,x)) (return 1)

test_pooling2d :: Spec
test_pooling2d = describe "HNN.Layers.pooling2d" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    check_backward_layer
      (pooling2d (Proxy :: Proxy [[2,2],[0,0],[2,2]]) pooling_max)
      (return $ HLS HNil)
      (normal 0 0.1 :: GPU (Tensor [1,1,4,4] CFloat))
      (normal 0 0.1 :: GPU (Tensor [1,1,2,2] CFloat))
