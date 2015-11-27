{-# LANGUAGE UndecidableInstances #-}
module Test.HNN.NumericGrad where

import Prelude hiding ((.), id)
import Foreign.C
import Control.Monad
import Control.Monad.Trans
import Data.VectorSpace
import GHC.TypeLits
import Data.Proxy

import HNN.NN
import HNN.Tensor

numericBwd :: forall a input_shape out_shape m
           . (TensorDataType a, Shape input_shape, Shape out_shape, Monad m)
           => (Tensor input_shape a -> m (Tensor out_shape a))
           -> Tensor input_shape a
           -> Tensor out_shape a
           -> m (Tensor input_shape a)
numericBwd f inp upgrad = do
  let inplist = toList inp
      upgradlist = toList upgrad
      insize = size (Proxy :: Proxy input_shape)
      h = 10E-5
      finitediff i = do
        fph <- f (shift i h)
        fmh <- f (shift i (-h))
        return $ (1/(2*h)) *^ (fph - fmh)
      shift i offset = fromList [if j /= i then inplist!!j else inplist!!j + offset | j <- [0..insize-1]]
  listGrad <- forM [0..insize-1] $ \i -> do
    fdiff <- finitediff i
    return $ fdiff <.> upgrad
  return $ fromList listGrad

genericNumericBwd :: (ToTensor inp, ToTensor out, Scalar inp ~ Scalar out,
                      KnownNat (ConcatSize out), KnownNat (ConcatSize inp),
                      Monad m, TensorDataType (Scalar inp))
                  => (inp -> m out)
                  -> inp
                  -> out
                  -> m inp
genericNumericBwd f inp upgrad = do
  let tinp = toTensor inp
      tupgrad = toTensor upgrad
      tf tx = do
        let x = fromTensor tx
        y <- f x
        return $ toTensor y
  tgrad <- numericBwd tf tinp tupgrad
  return $ fromTensor tgrad

class ToTensor t where
  type ConcatSize t :: Nat
  toTensor :: t -> (Tensor '[ConcatSize t] (Scalar t))
  fromTensor :: Tensor '[ConcatSize t] (Scalar t) -> t

instance (Shape s, KnownNat (Size s), TensorDataType a) => ToTensor (Tensor s a) where
  type ConcatSize (Tensor s a) = Size s
  toTensor = reshape
  fromTensor = reshape

instance ToTensor CFloat where
  type ConcatSize CFloat = 1
  toTensor x = fromList [x]
  fromTensor = head . toList

instance ToTensor CDouble where
  type ConcatSize CDouble = 1
  toTensor x = fromList [x]
  fromTensor = head . toList

instance (TensorDataType a) => ToTensor (HLSpace a '[]) where
  type ConcatSize (HLSpace a '[]) = 0
  toTensor _ = fromList []
  fromTensor _ = HLS HNil

instance forall e a l
         . (ToTensor e, ToTensor (HLSpace a l), a ~ Scalar e, a ~ Scalar (HLSpace a l),
            TensorDataType a, KnownNat (ConcatSize e),
            KnownNat (ConcatSize (HLSpace a l)),
            KnownNat (ConcatSize e + ConcatSize (HLSpace a l)))
         => ToTensor (HLSpace a (e ': l)) where
  type ConcatSize (HLSpace a (e ': l)) = ConcatSize e + ConcatSize (HLSpace a l)
  toTensor (HLS (HCons x xs)) = tconcat (toTensor x) (toTensor (HLS xs :: HLSpace a l))
  fromTensor t = let (t1,t2) = tsplit t
                 in HLS $ HCons (fromTensor t1) (unHLS (fromTensor t2 :: HLSpace a l))

instance (ToTensor a, ToTensor b, s ~ Scalar a, s ~ Scalar b, TensorDataType s,
          KnownNat (ConcatSize a), KnownNat (ConcatSize b),
          KnownNat (ConcatSize a + ConcatSize b))
         => ToTensor (a,b) where
  type ConcatSize (a,b) = ConcatSize a + ConcatSize b
  toTensor (a,b) = tconcat (toTensor a) (toTensor b)
  fromTensor t = let (t1,t2) = tsplit t
                 in (fromTensor t1, fromTensor t2)

