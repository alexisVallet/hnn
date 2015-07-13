{-# LANGUAGE GADTs, ScopedTypeVariables, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module HNN.NN where

import Prelude hiding ((.))
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

instance WeightsClass (Tensor a) a where
  toWeights tensor = NodeWeights [tensor]

instance WeightsClass [Tensor a] a where
  toWeights tensorlist = NodeWeights tensorlist

instance (WeightsClass w1 a, WeightsClass w2 a) => WeightsClass (w1, w2) a where
  toWeights (w1, w2) = Comp (toWeights w1) (toWeights w2)

data NN m a inp out where
  NN :: (WeightsClass w a)
     => (w -> inp -> m out) -- forward pass
     -> (w -> inp -> out -> out -> m (w, inp)) -- backward pass
     -> w -- current weights
     -> NN m a inp out

instance (Monad m) => Category (NN m a) where
  -- The identity function. Its backward pass just passes the gradient
  -- from upper layers similarly untouched.
  id = NN idfwd idbwd ()
       where idfwd () t = return t
             idbwd () _ _ outgrad = return ((), outgrad)
  -- Composes forward passes in the obvious manner, and intertwines
  -- the forward passes and backward passes of both to build the new
  -- backward pass.
  NN fwdbc bwdbc wbc . NN fwdab bwdab wab = NN fwdac bwdac (wbc, wab)
    where
      fwdac (w1, w2) a = fwdab w2 a >>= fwdbc w1
      bwdac (w1, w2) a out outgrad = do
        b <- fwdab w2 a
        (w1', b') <- bwdbc w1 b out outgrad
        (w2', a') <- bwdab w2 a b b'
        return ((w1', w2'), a')

-- Functors, catamorphisms and anamorphisms in the
-- neural net category
class (Category cat) => CFunctor cat f where
  cfmap :: (cat a b) -> (cat (f a) (f b))

type NNAlgebra m s f a = NN m s (f a) a

type NNCoAlgebra m s f a = NN m s a (f a)

newtype Fix f = Iso { invIso :: f (Fix f) }

-- essentially the id function, just here to satisfy ghc.
isoNN :: Monad m => NN m s (f (Fix f)) (Fix f)
isoNN = NN fwdiso bwdiso ()
  where fwdiso () = return . Iso
        bwdiso () _ _ fixf = return ((), invIso fixf)

invIsoNN :: Monad m => NN m s (Fix f) (f (Fix f))
invIsoNN = NN fwdinviso bwdinviso ()
  where fwdinviso () = return . invIso
        bwdinviso () _ _ ffixf = return ((), Iso ffixf)

cataNN :: (Monad m, CFunctor (NN m s) f)
       => NNAlgebra m s f a
       -> NN m s (Fix f) a
cataNN alg = alg . cfmap (cataNN alg) . invIsoNN

anaNN :: (Monad m, CFunctor (NN m s) f)
      => NNCoAlgebra m s f a
      -> NN m s a (Fix f)
anaNN coalg = isoNN . cfmap (anaNN coalg) . coalg
