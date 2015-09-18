{-# LANGUAGE GADTs, ScopedTypeVariables, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts, TypeFamilies #-}
module HNN.NN where

import Foreign.C
import Prelude hiding (id, (.))
import Control.Category
import Data.VectorSpace

import HNN.Tensor
import qualified HNN.NN.Mutable as M

-- Type machinery to make it a Category instance.
data Weights a = None
               | NodeWeights [Tensor a]
               | Comp (Weights a) (Weights a)

tensors :: Weights a -> [Tensor a]
tensors None = []
tensors (NodeWeights t) = t
tensors (Comp l r) = tensors l ++ tensors r

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
newtype NN m a w inp out = NN {
  -- Runs the forward pass (w -> inp -> m out), and additionally returns
  -- the corresponding backward pass (out -> (w, inp)) to run on some
  -- gradient from an upper layer, or just 1 in case this is the last
  -- layer.
  forwardBackward :: w -> inp -> m (out, out -> (w, inp))
  }

forward :: Monad m => NN m a w inp out -> w -> inp -> m out
forward nn w inp = do
  (out, _) <- forwardBackward nn w inp
  return out

-- Smart constructor from separate forward and backward pass
-- computations. Ensures that forward pass results will not be
-- recomputed needlessly during backward pass.
fromFwdBwd :: (Monad m)
           => (w -> inp -> m out) -- forward pass, takes weights and input and gives output
           -> (w -> inp -> out -> m (out -> (w, inp))) -- backward pass, takes weights, inputs, output from forward pass, gradient from upper layer, and outputs gradients for weights and inputs.
           -> NN m a w inp out
fromFwdBwd fwd bwd = NN fwdbwd
  where fwdbwd w inp = do
          fwdres <- fwd w inp
          bwdpure <- bwd w inp fwdres
          return (fwdres, bwdpure)

-- Composes forward passes keeping weights separate. This forms a category
-- structure, but cannot be made an instance of the standard Category type class
-- without discarding type information about the weights. It is obviously useful
-- to build regular feed-forward neural nets, but other categorical constructs
-- seem weird. Catamorphisms and Anamorphisms basically turn this into infinite
-- depth neural networks. Haven't found a way to make GHC deal with the types
-- yet, not sure it can deal with it at runtime either, but it's an intriguing
-- theoretical construct.
(<+>) :: Monad m => NN m s w1 b c -> NN m s w2 a b -> NN m s (w1, w2) a c
NN fwdbwdbc <+> NN fwdbwdab = NN fwdbwdac
  where fwdbwdac (w1,w2) a = do
          (fwdabres, bwdab) <- fwdbwdab w2 a
          (fwdbcres, bwdbc) <- fwdbwdbc w1 fwdabres
          return (fwdbcres, \cgrad -> let (w1grad, bgrad) = bwdbc cgrad
                                          (w2grad, agrad) = bwdab bgrad in
                                      ((w1grad, w2grad), agrad))

-- We require weights to form a vector space over the tensor scalar datatype.
-- This category instance requires weights to be shared between composed
-- layers. So they should have the same weight type, and individual tensors
-- making up this weight type should have the same shape.
-- This Category instance composes neural nets so they share weights - as
-- in the successive applications of the same layer in a recurrent network.
instance (Monad m, VectorSpace w) => Category (NN m s w) where
  id = fromFwdBwd idFwd idBwd
    where idFwd _ inp = return inp
          idBwd _ _ _ = return $ \upgrad -> (zeroV, upgrad)
  NN fwdbwdbc . NN fwdbwdab = NN fwdbwdac
    where fwdbwdac w a = do
            (fwdabres, bwdab) <- fwdbwdab w a
            (fwdbcres, bwdbc) <- fwdbwdbc w fwdabres
            return (fwdbcres, \cgrad -> let (wgrad1, bgrad) = bwdbc cgrad
                                            (wgrad2, agrad) = bwdab bgrad in
                                        (wgrad1 ^+^ wgrad2, agrad))

-- For convenience, we define trivial vector spaces for layers which do
-- not have any weights. The parameter is for the corresponding scalar type.
data Trivial a = Zero

instance AdditiveGroup (Trivial a) where
  zeroV = Zero
  Zero ^+^ Zero = Zero
  negateV Zero = Zero

instance VectorSpace (Trivial a) where
  type Scalar (Trivial a) = a
  s *^ Zero = Zero

-- Functors, catamorphisms and anamorphisms in the
-- neural net categories.
class NNFunctor f where
  nnfmap :: (NN m a w inp out) -> (NN m a w (f inp) (f out))

type NNAlgebra m s w f a = NN m s w (f a) a

type NNCoAlgebra m s w f a = NN m s w a (f a)

newtype Fix f = Iso { invIso :: f (Fix f) }

-- essentially the id function, just here to satisfy ghc.
isoNN :: (Monad m, VectorSpace w) => NN m s w (f (Fix f)) (Fix f)
isoNN = fromFwdBwd fwdiso bwdiso
  where fwdiso _ = return . Iso
        bwdiso _ _ _ = return $ \fixf -> (zeroV, invIso fixf) 

invIsoNN :: (Monad m, VectorSpace w) => NN m s w (Fix f) (f (Fix f))
invIsoNN = fromFwdBwd fwdinviso bwdinviso
  where fwdinviso _ = return . invIso
        bwdinviso _ _ _ = return $ \ffixf -> (zeroV, Iso ffixf)

cataNN :: (Monad m, NNFunctor f, VectorSpace w)
       => NNAlgebra m s w f a
       -> NN m s w (Fix f) a
cataNN alg = alg . nnfmap (cataNN alg) . invIsoNN

anaNN :: (Monad m, NNFunctor f, VectorSpace w)
      => NNCoAlgebra m s w f a
      -> NN m s w a (Fix f)
anaNN coalg = isoNN . nnfmap (anaNN coalg) . coalg
