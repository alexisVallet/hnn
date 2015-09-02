{-# LANGUAGE GADTs, ScopedTypeVariables, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
module HNN.NN where

import Prelude hiding (id, (.))
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

-- We fuse forward pass and backward pass into a single function
-- as an optimization to avoid unnecessary recomputation of forward
-- passes (if properly implemented). fromFwdBwd demonstrates the trick.
data NN m a inp out where
  NN :: (WeightsClass w a)
     => (w -> inp -> m (out, out -> (w, inp))) -- forward pass and backward pass
     -> w -- current weights
     -> NN m a inp out

-- Smart constructor from separate forward and backward pass
-- computations. Ensures that forward pass results will not be
-- recomputed needlessly during backward pass.
fromFwdBwd :: (WeightsClass w a, Monad m)
           => (w -> inp -> m out) -- forward pass, takes weights and input and gives output
           -> (w -> inp -> out -> m (out -> (w, inp))) -- backward pass, takes weights, inputs, output from forward pass, gradient from upper layer, and outputs gradients for weights and inputs.
           -> w
           -> NN m a inp out
fromFwdBwd fwd bwd w = NN fwdbwd w
  where fwdbwd w' inp = do
          fwdres <- fwd w' inp
          bwdpure <- bwd w' inp fwdres
          return (fwdres, bwdpure)

instance (Monad m) => Category (NN m a) where
  -- The identity function. Its backward pass just passes the gradient
  -- from upper layers similarly untouched.
  id = fromFwdBwd idfwd idbwd ()
       where idfwd () t = return t
             idbwd () _ _ = return $ \x -> ((), x)
  -- Composes forward passes in the obvious manner, and intertwines
  -- the forward passes and backward passes of both to build the new
  -- backward pass.
  NN fwdbwdbc w1 . NN fwdbwdab w2 = NN fwdbwdac (w1, w2)
    where fwdbwdac (w1',w2') a = do
            (fwdabres, bwdab) <- fwdbwdab w2' a
            (fwdbcres, bwdbc) <- fwdbwdbc w1' fwdabres
            return (fwdbcres, \cgrad -> let (w1grad, bgrad) = bwdbc cgrad
                                            (w2grad, agrad) = bwdab bgrad in
                                        ((w1grad, w2grad), agrad))

-- Functors, catamorphisms and anamorphisms in the
-- neural net category
class (Category cat) => CFunctor cat f where
  cfmap :: (cat a b) -> (cat (f a) (f b))

type NNAlgebra m s f a = NN m s (f a) a

type NNCoAlgebra m s f a = NN m s a (f a)

newtype Fix f = Iso { invIso :: f (Fix f) }

-- essentially the id function, just here to satisfy ghc.
isoNN :: Monad m => NN m s (f (Fix f)) (Fix f)
isoNN = fromFwdBwd fwdiso bwdiso ()
  where fwdiso () = return . Iso
        bwdiso () _ _ = return $ \fixf -> ((), invIso fixf) 

invIsoNN :: Monad m => NN m s (Fix f) (f (Fix f))
invIsoNN = fromFwdBwd fwdinviso bwdinviso ()
  where fwdinviso () = return . invIso
        bwdinviso () _ _ = return $ \ffixf -> ((), Iso ffixf)

cataNN :: (Monad m, CFunctor (NN m s) f)
       => NNAlgebra m s f a
       -> NN m s (Fix f) a
cataNN alg = alg . cfmap (cataNN alg) . invIsoNN

anaNN :: (Monad m, CFunctor (NN m s) f)
      => NNCoAlgebra m s f a
      -> NN m s a (Fix f)
anaNN coalg = isoNN . cfmap (anaNN coalg) . coalg

-- Actually running a neural net.
forward :: (Monad m) => NN m a inp out -> inp -> m out
forward (NN fwdbwd w) inp = do
  (out, _) <- fwdbwd w inp
  return out
