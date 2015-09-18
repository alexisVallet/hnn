{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module Test.HNN.Layers where

import Foreign.C
import Test.Hspec
import Data.VectorSpace
import qualified Numeric.LinearAlgebra.HMatrix as HMatrix

import HNN.NN
import HNN.Layers
import HNN.Tensor

test_dropout :: Spec
test_dropout = describe "HNN.Layers.dropout" $ do
  let input = fromList [1,1,8,8] [1..8*8] :: Tensor CFloat
  it "does nothing for drop_proba = 0" $ do
    (fmap toList $ runLayer 42 $ forward (dropout 0) Zero input)
     `shouldReturn` toList input
  it "returns all zeros for drop_proba = 1" $ do
    (fmap toList $ runLayer 42 $ forward (dropout 1) Zero input)
     `shouldReturn` (take (8*8) $ repeat 0)

test_linear :: Spec
test_linear = describe "HNN.Layers.linear" $ do
  let x = fromList [2,2] [1,2,3,4] :: Tensor CFloat
      y = fromList [2,2] [5,6,7,8] :: Tensor CFloat
      expected = [19,22,43,50] :: [CFloat]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runLayer 42 $ forward linear y x)
     `shouldReturn` expected

test_sumCols :: Spec
test_sumCols = describe "HNN.Layers.sumCols" $ do
  let x = fromList [2,2] [1,2,3,4] :: Tensor CFloat
      expected = [4,6] :: [CFloat]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runLayer 42 $ forward sumCols Zero x)
     `shouldReturn` expected

test_sumRows :: Spec
test_sumRows = describe "HNN.Layers.sumRows" $ do
  let x = fromList [2,2] [1,2,3,4] :: Tensor CFloat
      expected = [3,7] :: [CFloat]
  it "returns the right result for a simple example" $ do
    (fmap toList $ runLayer 42 $ forward sumRows Zero x)
     `shouldReturn` expected  
