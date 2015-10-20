{-# LANGUAGE TemplateHaskell #-}
module HNN.Optimize.Vanilla (
    VanillaT
  , runVanilla
  , vanilla
  ) where

import Control.Monad.Reader
import Data.VectorSpace
import Control.Lens

data VanillaReader s = VanillaReader {
  _learningRate :: s
  }
makeLenses ''VanillaReader

type VanillaT s = ReaderT (VanillaReader s)

runVanilla learningRate action =
  runReaderT action (VanillaReader learningRate)

vanilla :: (VectorSpace w, Monad m) => Scalar w -> w -> w -> VanillaT (Scalar w) m w
vanilla cost grad w_t = do
  lr <- view learningRate
  return $ w_t ^-^ lr *^ grad
