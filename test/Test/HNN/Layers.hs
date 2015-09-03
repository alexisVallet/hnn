module Test.HNN.Layers where

import Foreign.C
import Test.Hspec
import qualified Numeric.LinearAlgebra.HMatrix as HMatrix

import HNN.NN
import HNN.Layers
import HNN.Tensor

import Test.HNN.NN.NumericGrad

-- recursively flattens a set of weights for a neural net
-- on the GPU side into a single CPU side vector.
weights_to_vector :: (TensorDataType a, WeightsClass w a) => w -> HMatrix.Vector a
weights_to_vector = HMatrix.vjoin . map toVec . tensors . toWeights
  where toVec = HMatrix.fromList . toList

test_dropout :: Spec
test_dropout = describe "HNN.Layers.dropout" $ do
  let input = fromList [1,1,8,8] [1..8*8] :: Tensor CFloat
  it "does nothing for drop_proba = 0" $ do
    (fmap toList $ runLayer 42 $ forward (dropout 0) Zero input)
     `shouldReturn` toList input
  it "returns all zeros for drop_proba = 1" $ do
    (fmap toList $ runLayer 42 $ forward (dropout 1) Zero input)
     `shouldReturn` (take (8*8) $ repeat 0)
