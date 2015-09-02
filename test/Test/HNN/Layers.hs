module Test.HNN.Layers where

import Foreign.C
import Test.Hspec

import HNN.NN
import HNN.Layers
import HNN.Tensor

test_dropout :: Spec
test_dropout = describe "HNN.Layers.dropout" $ do
  let input = fromList [1,1,8,8] [1..8*8] :: Tensor CFloat
  it "does nothing for drop_proba = 0" $ do
    (fmap toList $ runLayer 42 $ forward (dropout 0) input) `shouldReturn` toList input
  it "returns all zeros for drop_proba = 1" $ do
    (fmap toList $ runLayer 42 $ forward (dropout 1) input) `shouldReturn` (take (8*8) $ repeat 0)
