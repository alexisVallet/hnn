{-# LANGUAGE GADTs, ScopedTypeVariables, MultiParamTypeClasses, FlexibleInstances, InstanceSigs #-}
module HNN.NN where

import Control.Category

import HNN.Tensor
import qualified HNN.NN.Mutable as M

-- Type machinery to make it a Category instance.
data Weights a = None
               | NodeWeights [Tensor a]
               | Comp (Weights a) (Weights a)

class WeightsClass w a where
  toWeights :: w -> Weights a

instance WeightsClass () a where
  toWeights () = None

instance WeightsClass [Tensor a] a where
  toWeights tensorlist = NodeWeights tensorlist

instance (WeightsClass w1 a, WeightsClass w2 a) => WeightsClass (w1, w2) a where
  toWeights (w1, w2) = Comp (toWeights w1) (toWeights w2)

data NN m a inp out where
  NN :: (WeightsClass w a)
     => (w -> inp -> m out) -- forward pass
     -> (w -> inp -> out -> m (w, inp)) -- backward pass
     -> NN m a inp out

instance (Monad m) => Category (NN m a) where
  -- The identity function. Its backward pass just passes the gradient
  -- from upper layers untouched - after all, it does precisely nothing.
  id :: NN m a inp inp
  id = NN idfwd idbwd
       where idfwd () t = return t
             idbwd () _ outgrad = return ((), outgrad)
  -- Composes forward passes in the obvious manner, and intertwines
  -- the forward passes and backward passes of both to build the new
  -- backward pass.
  NN fwdbc bwdbc . NN fwdab bwdab = NN fwdac bwdac
    where
      fwdac (w1, w2) a = fwdab w2 a >>= fwdbc w1
      bwdac (w1, w2) a c = do
        b <- fwdab w2 a
        (w1', b') <- bwdbc w1 b c
        (w2', a') <- bwdab w2 a b'
        return ((w1', w2'), a')
