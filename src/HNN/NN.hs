{-# LANGUAGE GADTs, ScopedTypeVariables, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts, TypeFamilies, DataKinds, TypeOperators, DeriveGeneric #-}
module HNN.NN (
    module Control.Category
  , module Data.VectorSpace
  , Diff(..)
  , NNCategory(..)
  , Differentiable(..)
  , (-<)
  , fromFwdBwd
  , Trivial(..)
  , Layer(..)
  , (<+<)
  , (>+>)
  , noWeights
  , effect
  ) where

import Foreign.C
import Prelude hiding (id, (.))
import Control.Category
import Data.VectorSpace
import GHC.Generics

import HNN.Tensor
import qualified HNN.NN.Mutable as M

-- Represents a computation from a to b that is differentiable with regards to a.
-- The inner computation can take place in a monad, for instance for random number
-- generation.
newtype Diff m s a b = Diff (a -> m (b, b -> a))

instance (Monad m) => Category (Diff m s) where
  id = Diff $ \a -> return (a, \x -> x)
  Diff fbc . Diff fab = Diff $ \a -> do
    (b, bwdab) <- fab a
    (c, bwdbc) <- fbc b
    return (c, bwdab . bwdbc)

infixr 3 ***
infixr 3 &&&
class (Category c) => NNCategory c where
  (***) :: (VectorSpace a, VectorSpace a', Scalar a ~ Scalar a')
        => c a b -> c a' b' -> c (a,a') (b,b')
  (&&&) :: (VectorSpace a)
        => c a b -> c a b' -> c a (b,b')
  terminal :: (VectorSpace a) => b -> c a b

instance (Monad m) => NNCategory (Diff m s) where
  Diff f1 *** Diff f2 = Diff $ \(i1,i2) -> do
    (o1, bwd1) <- f1 i1
    (o2, bwd2) <- f2 i2
    return ((o1,o2), \(o1',o2') -> (bwd1 o1', bwd2 o2')) 
  Diff f1 &&& Diff f2 = Diff $ \x -> do
    (out1, bwd1) <- f1 x
    (out2, bwd2) <- f2 x
    return ((out1,out2), \(y1,y2) -> bwd1 y1 ^+^ bwd2 y2)
  terminal x = fromFwdBwd constFwd constBwd
    where constFwd _ = return x
          constBwd _ _ = return $ const zeroV

class (Monad (DiffMonad f)) => Differentiable f where
  type Input f :: *
  type Output f :: *
  type DiffMonad f :: * -> *
  forwardBackward :: f -> Input f -> DiffMonad f (Output f, Output f -> Input f)
  forward :: f -> Input f -> DiffMonad f (Output f)
  forward df inp = do
    (out, _) <- forwardBackward df inp
    return out
  backward :: f -> Input f -> Output f -> DiffMonad f (Input f)
  backward df inp upgrad = do
    (out, bwd) <- forwardBackward df inp
    return $ bwd upgrad

infixr 2 -<
(-<) :: (VectorSpace i1, VectorSpace i2, NNCategory c) => c (i1,i2) out -> i1 -> c i2 out
nn -< inp = terminal inp &&& id >>> nn


instance (Monad m) => Differentiable (Diff m s a b) where
  type Input (Diff m s a b) = a
  type Output (Diff m s a b) = b
  type DiffMonad (Diff m s a b) = m
  forwardBackward (Diff f) = f

-- Smart constructor from separate forward and backward pass
-- computations. Ensures that forward pass results will not be
-- recomputed needlessly during backward pass.
fromFwdBwd :: (Monad m, VectorSpace inp)
           => (inp -> m out) -- forward pass, takes weights and input and gives output
           -> (inp -> out -> m (out -> inp))
           -> Diff m a inp out
fromFwdBwd fwd bwd = Diff $ \inp -> do
  fwdres <- fwd inp
  bwdpure <- bwd inp fwdres
  return (fwdres, bwdpure)

-- Vector spaces as heterogeneous lists of vector spaces. Just an heterogeneous
-- list with the scalar type as a phantom type.
data Trivial a = Zero
               deriving (Generic, Show)

instance AdditiveGroup (Trivial a) where
  zeroV = Zero
  Zero ^+^ Zero = Zero
  negateV Zero = Zero

instance VectorSpace (Trivial a) where
  type Scalar (Trivial a) = a
  s *^ Zero = Zero

instance (AdditiveGroup a) => InnerSpace (Trivial a) where
  Zero <.> Zero = zeroV

-- A weight layer, as per common usage, has a distinguished set of weights.
newtype Layer m a w inp out = Layer {
  unLayer :: Diff m a (w,inp) out
  }

-- Category instance fits recurrent composition. (shared weights)
instance (VectorSpace w, Monad m) => Category (Layer m a w) where
  id = Layer $ Diff (\(w,x) -> return (x, \x' -> (zeroV, x')))
  Layer (Diff fbc) . Layer (Diff fab) = Layer $ Diff $ \(w,a) -> do
    (b, bwdab) <- fab (w,a)
    (c, bwdbc) <- fbc (w,b)
    return $ (c, \c' -> let (wgrad1,bgrad) = bwdbc c'
                            (wgrad2,agrad) = bwdab bgrad in
                        (wgrad1 ^+^ wgrad2, agrad))

instance (VectorSpace w, a ~ Scalar w, Monad m) => NNCategory (Layer m a w) where
  Layer (Diff f1) *** Layer (Diff f2) = Layer $ Diff $ \(w,(a,a')) -> do
    (b, bwd1) <- f1 (w,a)
    (b', bwd2) <- f2 (w,a')
    return ((b,b'), \(bgrad,bgrad') -> let (w1grad, agrad) = bwd1 bgrad
                                           (w2grad, agrad') = bwd2 bgrad'
                                       in (w1grad ^+^ w2grad, (agrad, agrad')))
  Layer (Diff f1) &&& Layer (Diff f2) = Layer $  Diff $ \(w,a) -> do
    (b,bwd1) <- f1 (w,a)
    (b',bwd2) <- f2 (w,a)
    return ((b,b'), \(bgrad,bgrad') -> let (w1grad, agrad1) = bwd1 bgrad
                                           (w2grad, agrad2) = bwd2 bgrad'
                                       in (w1grad ^+^ w2grad, agrad1 ^+^ agrad2))
  terminal x = Layer $ Diff $ const $ return (x, \_ -> zeroV)

instance (Monad m) => Differentiable (Layer m s w a b) where
  type Input (Layer m s w a b) = (w,a)
  type Output (Layer m s w a b) = b
  type DiffMonad (Layer m s w a b) = m
  forwardBackward (Layer df) = forwardBackward df

-- Ad-hoc combinator for feed-forward composition.
infixr 2 <+<
(<+<) :: (Monad m, VectorSpace w1, VectorSpace w2, s ~ Scalar w1, s ~ Scalar w2)
      => Layer m s w1 b c -> Layer m s w2 a b -> Layer m s (w1, w2) a c
Layer (Diff fbc) <+< Layer (Diff fab) = Layer $ Diff $ \((w1,w2),a) -> do
  (b, bwdab) <- fab (w2,a)
  (c, bwdbc) <- fbc (w1,b)
  return (c, \c' -> let (w1grad, bgrad) = bwdbc c'
                        (w2grad, agrad) = bwdab bgrad in
                     ((w1grad,w2grad), agrad))

infixl 2 >+>
(>+>) :: forall m w1 w2 n s a b c .
         (Monad m, VectorSpace w1, VectorSpace w2, s ~ Scalar w1, s ~ Scalar w2)
      => Layer m s w2 a b -> Layer m s w1 b c -> Layer m s (w1,w2) a c
f >+> g = g <+< f

-- Making a layer out of a differentiable function that does not depend on a set
-- of weights.
noWeights :: (Monad m, VectorSpace w, a ~ Scalar w)
          => Diff m a inp out -> Layer m a w inp out
noWeights (Diff f) = Layer $ Diff $ \(w,x) -> do
  (y, bwd) <- f x
  return (y, \y' -> (zeroV, bwd y'))

-- Layer that runs an effectful computation on the input and passes it along
-- untouched. Useful for debugging.
effect :: (Monad m, VectorSpace w, s ~ Scalar w) => (a -> m b) -> Layer m s w a a
effect f = noWeights $ Diff $ \x -> do
  f x
  return (x, \x' -> x')
