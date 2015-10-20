{-# LANGUAGE FlexibleContexts, TemplateHaskell #-}
module HNN.Optimize (
    module HNN.Optimize.Momentum
  , module HNN.Optimize.Vanilla
  , sgd
  ) where
import Control.Monad.Morph
import Control.Monad.Trans
import Control.Monad.State
import Data.VectorSpace

import HNN.Optimize.Momentum
import HNN.Optimize.Vanilla

import Pipes
import Pipes.Lift

sgd :: (Monad m, MonadTrans t, Monad (t (Pipe a (Scalar w, w) m)), Monad (t m), MFunctor t, VectorSpace w)
    => (Scalar w -> w -> w -> t m w) -- update function
    -> (w -> a -> m (Scalar w, w)) -- cost function gradient
    -> w -- initial weights
    -> t (Pipe a (Scalar w, w) m) () -- streams out cost weights for each iteration
sgd update cost_grad w_0 =
  distribute $ sgd' update cost_grad w_0

sgd' :: (Monad m, MonadTrans t, Monad (t m), VectorSpace w)
     => (Scalar w -> w -> w -> t m w) -- update function
     -> (w -> a -> m (Scalar w, w)) -- cost function gradient
     -> w -- initial weights
     -> Pipe a (Scalar w, w) (t m) () -- streams out cost weights for each iteration
sgd' update cost_grad w_0 = evalStateT action w_0 
  where action = forever $ do
          w_t <- get
          batch <- lift $ await
          (cost, grad) <- lift $ lift $ lift $ cost_grad w_t batch
          w_tp1 <- lift $ lift $ update cost grad w_t
          lift $ yield (cost, w_tp1)
          put w_tp1
