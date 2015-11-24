{-# LANGUAGE GADTs, ScopedTypeVariables, MultiParamTypeClasses, FlexibleInstances, FlexibleContexts, TypeFamilies, DataKinds, TypeOperators, DeriveGeneric, InstanceSigs #-}
module HNN.NN (
    module Control.Category
  , module Data.VectorSpace
  , Diff(..)
  , NNCategory(..)
  , Differentiable(..)
  , (-<)
  , fromFwdBwd
  , Trivial(..)
  , HLSpace(..)
  , space
  , HList(..)
  , Layer(..)
  , FeedForward(..)
  , simple
  , noWeights
  , effect
  ) where

import Foreign.C
import Prelude hiding (id, (.))
import Control.Category
import Data.VectorSpace
import GHC.Generics
import Data.HList.HList
import Data.HList.HListPrelude
import Data.Proxy
import Data.Serialize
import Control.DeepSeq

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

infixr 4 ***
infixr 4 &&&
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

infixr 4 -<
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
fromFwdBwd :: (Monad m)
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

-- We store weights in heterogeneous lists internally, which get concatenated
-- by composition.
newtype HLSpace a l = HLS {
  unHLS :: HList l
  }

space :: (TensorDataType a) => Proxy a -> HList l -> HLSpace a l
space _ = HLS

-- Additive group instance for list of weights.
instance AdditiveGroup (HLSpace a '[]) where
  HLS HNil ^+^ HLS HNil = HLS HNil
  zeroV = HLS HNil
  negateV (HLS HNil) = HLS HNil

instance forall a e (l :: [*])
         . (AdditiveGroup e, AdditiveGroup (HLSpace a l))
         => AdditiveGroup (HLSpace a (e ': l)) where
  HLS (HCons x1 xs1) ^+^ HLS (HCons x2 xs2) =
    HLS $ HCons (x1 ^+^ x2) (unHLS (HLS xs1 ^+^ HLS xs2 :: HLSpace a l))
  zeroV = HLS $ HCons zeroV (unHLS (zeroV :: HLSpace a l))
  negateV (HLS (HCons x xs)) = HLS $ negateV x `HCons` unHLS (negateV (HLS xs :: HLSpace a l))
                                      
-- Vector space instance for list of weights.
instance VectorSpace (HLSpace a '[]) where
  type Scalar (HLSpace a '[]) = a
  x *^ HLS HNil = HLS HNil

instance forall a e (l :: [*])
         . (VectorSpace e, VectorSpace (HLSpace a l), a ~ Scalar e,
            a ~ Scalar (HLSpace a l))
         => VectorSpace (HLSpace a (e ': l)) where
  type Scalar (HLSpace a (e ': l)) = a
  s *^ HLS (HCons x xs) = HLS $ HCons (s *^ x) (unHLS (s *^ HLS xs :: HLSpace a l))

-- Serializable heterogeneous lists of weights.
instance Serialize (HLSpace a '[]) where
  put (HLS HNil) = return ()
  get = return (HLS HNil)
                    
instance forall a e (l :: [*])
         . (Serialize e, Serialize (HLSpace a l))
         => Serialize (HLSpace a (e ': l)) where
  put (HLS (HCons x xs)) = do
    put x
    put (HLS xs :: HLSpace a l)
  get = do
    x <- (get :: Get e)
    HLS xs <- (get :: Get (HLSpace a l))
    return (HLS $ x `HCons` xs)

-- Deepseqable heterogeneous lists of weights.
instance NFData (HLSpace a '[]) where
  rnf (HLS HNil) = ()

instance forall e a l
         . (NFData e, NFData (HLSpace a l))
         => NFData (HLSpace a (e ': l)) where
  rnf (HLS (HCons x xs)) = deepseq (x, HLS xs :: HLSpace a l) ()

-- Elementwise numeric instances for weights.
instance Num (HLSpace a '[]) where
  HLS HNil + HLS HNil = HLS HNil
  HLS HNil - HLS HNil = HLS HNil
  HLS HNil * HLS HNil = HLS HNil
  abs (HLS HNil) = HLS HNil
  signum (HLS HNil) = HLS HNil
  fromInteger _ = HLS HNil

instance forall a e (l :: [*])
         . (Num e, Num (HLSpace a l))
         => Num (HLSpace a (e ': l)) where
  HLS (HCons x1 xs1) + HLS (HCons x2 xs2) =
     HLS $ HCons (x1 + x2) (unHLS (HLS xs1 + HLS xs2 :: HLSpace a l))
  HLS (HCons x1 xs1) - HLS (HCons x2 xs2) =
     HLS $ HCons (x1 - x2) (unHLS (HLS xs1 - HLS xs2 :: HLSpace a l))
  HLS (HCons x1 xs1) * HLS (HCons x2 xs2) =
     HLS $ HCons (x1 * x2) (unHLS (HLS xs1 * HLS xs2 :: HLSpace a l))
  abs (HLS (HCons x xs)) = HLS (HCons (abs x) (unHLS (abs $ HLS xs :: HLSpace a l)))
  signum (HLS (HCons x xs)) = HLS (HCons (signum x) (unHLS (signum $ HLS xs :: HLSpace a l)))
  fromInteger i = HLS (HCons (fromInteger i) (unHLS (fromInteger i :: HLSpace a l)))

instance Fractional (HLSpace a '[]) where
  recip (HLS HNil) = HLS HNil
  fromRational _ = HLS HNil

instance forall a e (l :: [*])
         . (Fractional e, Fractional (HLSpace a l))
         => Fractional (HLSpace a (e ': l)) where
  HLS (HCons x1 xs1) / HLS (HCons x2 xs2) =
     HLS $ HCons (x1 / x2) (unHLS (HLS xs1 / HLS xs2 :: HLSpace a l))
  recip (HLS (HCons x xs)) = HLS (HCons (recip x) (unHLS (recip $ HLS xs :: HLSpace a l)))
  fromRational r = HLS (HCons (fromRational r) (unHLS (fromRational r :: HLSpace a l)))

instance Floating (HLSpace a '[]) where
  pi = HLS HNil
  exp = id
  log = id
  sin = id
  cos = id
  asin = id
  acos = id
  atan = id
  sinh = id
  cosh = id
  tanh = id
  asinh = id
  acosh = id
  atanh = id

instance forall a e (l :: [*])
         . (Floating e, Floating (HLSpace a l))
         => Floating (HLSpace a (e ': l)) where
  pi = HLS (HCons pi (unHLS (pi :: HLSpace a l)))
  exp (HLS (HCons x xs)) = HLS (HCons (exp x) (unHLS (exp (HLS xs) :: HLSpace a l)))
  log (HLS (HCons x xs)) = HLS (HCons (log x) (unHLS (log (HLS xs) :: HLSpace a l)))
  sqrt (HLS (HCons x xs)) = HLS (HCons (sqrt x) (unHLS (sqrt (HLS xs) :: HLSpace a l)))
  sin (HLS (HCons x xs)) = HLS (HCons (sin x) (unHLS (sin (HLS xs) :: HLSpace a l)))
  cos (HLS (HCons x xs)) = HLS (HCons (cos x) (unHLS (cos (HLS xs) :: HLSpace a l)))
  tan (HLS (HCons x xs)) = HLS (HCons (tan x) (unHLS (tan (HLS xs) :: HLSpace a l)))
  asin (HLS (HCons x xs)) = HLS (HCons (asin x) (unHLS (asin (HLS xs) :: HLSpace a l)))
  acos (HLS (HCons x xs)) = HLS (HCons (acos x) (unHLS (acos (HLS xs) :: HLSpace a l)))
  atan (HLS (HCons x xs)) = HLS (HCons (atan x) (unHLS (atan (HLS xs) :: HLSpace a l)))
  sinh (HLS (HCons x xs)) = HLS (HCons (sinh x) (unHLS (sinh (HLS xs) :: HLSpace a l)))
  cosh (HLS (HCons x xs)) = HLS (HCons (cosh x) (unHLS (cosh (HLS xs) :: HLSpace a l)))
  tanh (HLS (HCons x xs)) = HLS (HCons (tanh x) (unHLS (tanh (HLS xs) :: HLSpace a l)))
  asinh (HLS (HCons x xs)) = HLS (HCons (asinh x) (unHLS (asinh (HLS xs) :: HLSpace a l)))
  acosh (HLS (HCons x xs)) = HLS (HCons (acosh x) (unHLS (acosh (HLS xs) :: HLSpace a l)))
  atanh (HLS (HCons x xs)) = HLS (HCons (atanh x) (unHLS (atanh (HLS xs) :: HLSpace a l)))
  HLS (HCons x1 xs1) ** HLS (HCons x2 xs2) =
    HLS $ HCons (x1**x2) (unHLS (HLS xs1 ** HLS xs2 :: HLSpace a l))
  logBase (HLS (HCons x1 xs1)) (HLS (HCons x2 xs2)) =
    HLS $ HCons (logBase x1 x2) (unHLS (logBase (HLS xs1) (HLS xs2) :: HLSpace a l))

-- For user simplicity, we make an interface converting from/to tuples
-- when possible.
class HasTupleRep t where
  type TupleRep t :: *
  toTuple :: t -> TupleRep t
  fromTuple :: TupleRep t -> t

-- A weight layer, as per common usage, has a distinguished set of weights.
newtype Layer m a (w :: [*]) inp out = Layer {
  unLayer :: Diff m a (HLSpace a w,inp) out
  }

-- Category instance fits recurrent composition. (shared weights)
instance forall a (w :: [*]) m . (VectorSpace (HLSpace a w), Monad m)
         => Category (Layer m a w) where
  id = Layer $ Diff (\(w,x) -> return (x, \x' -> (zeroV, x')))
  Layer (Diff fbc) . Layer (Diff fab) = Layer $ Diff $ \(w,a) -> do
    (b, bwdab) <- fab (w,a)
    (c, bwdbc) <- fbc (w,b)
    return $ (c, \c' -> let (wgrad1,bgrad) = bwdbc c'
                            (wgrad2,agrad) = bwdab bgrad in
                        (wgrad1 ^+^ wgrad2, agrad))

instance forall m a (w :: [*])
         . (VectorSpace (HLSpace a w), a ~ Scalar (HLSpace a w), Monad m)
         => NNCategory (Layer m a w) where
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
  type Input (Layer m s w a b) = (HLSpace s w,a)
  type Output (Layer m s w a b) = b
  type DiffMonad (Layer m s w a b) = m
  forwardBackward (Layer df) = forwardBackward df

-- Feed-forward composition.
infixr 3 >+>
class FeedForward (l :: (* -> *) -> * -> [*] -> * -> * -> *) where
  (>+>) :: forall s (w1 :: [*]) (w2 :: [*]) a b c m n
        . (Monad m, VectorSpace (HLSpace s w1), VectorSpace (HLSpace s w2),
           s ~ Scalar (HLSpace s w1), s ~ Scalar (HLSpace s w2),
           HAppendList w1 w2, HSplitAt n (HAppendListR w1 w2) w1 w2)
        => l m s w1 a b -> l m s w2 b c -> l m s (HAppendListR w1 w2) a c

-- For simple layers.
instance FeedForward Layer where
  (>+>) :: forall s (w1 :: [*]) (w2 :: [*]) a b c m n
        . (Monad m, VectorSpace (HLSpace s w1), VectorSpace (HLSpace s w2),
           s ~ Scalar (HLSpace s w1), s ~ Scalar (HLSpace s w2),
           HAppendList w1 w2, HSplitAt n (HAppendListR w1 w2) w1 w2)
        => Layer m s w1 a b -> Layer m s w2 b c -> Layer m s (HAppendListR w1 w2) a c
  Layer (Diff fab) >+> Layer (Diff fbc) = Layer $ Diff $ \(w1w2,a) -> do
    let (w1,w2) = hSplitAt (Proxy :: Proxy n) $ unHLS w1w2
    (b, bwdab) <- fab (HLS w1,a)
    (c, bwdbc) <- fbc (HLS w2,b)
    return (c, \c' -> let (w2grad, bgrad) = bwdbc c'
                          (w1grad, agrad) = bwdab bgrad in
                      (HLS $ hAppendList (unHLS w1grad) (unHLS w2grad), agrad))

-- Making a layer out of a differentiable function that does not depend on a set
-- of weights.
noWeights :: (Monad m)
          => Diff m a inp out -> Layer m a '[] inp out
noWeights (Diff f) = Layer $ Diff $ \(_,x) -> do
  (y, bwd) <- f x
  return (y, \y' -> (zeroV, bwd y'))

-- Layer that runs an effectful computation on the input and passes it along
-- untouched. Useful for debugging.
effect :: (Monad m)
       => (a -> m b) -> Layer m s '[] a a
effect f = noWeights $ Diff $ \x -> do
  f x
  return (x, \x' -> x')

-- Specifying a layer alongside its initial weights.
data Init m s (w :: [*]) a b = Init {
  layer :: Layer m s w a b,
  initWeights :: HLSpace a w
  }

-- Special case for no weights.
simple :: Layer m s '[] a b -> Init m s '[] a b
simple l = Init l (HLS HNil)

-- For layers with specified initial weights.
instance FeedForward Init where
  Init l1 w1 >+> Init l2 w2 = Init (l1 >+> l2) (HLS $ hAppendList (unHLS w1) (unHLS w2))
