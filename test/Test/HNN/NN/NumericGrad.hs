{-# LANGUAGE FlexibleContexts, TypeFamilies, GADTs #-}
module Test.HNN.NN.NumericGrad where

import Foreign.C
import Control.Monad
import Control.Monad.Trans
import Data.VectorSpace

import HNN.NN
import HNN.Tensor

numericBwd :: (TensorDataType a, Monad m)
           => (Tensor a -> m (Tensor a))
           -> Tensor a
           -> Tensor a
           -> m (Tensor a)
numericBwd f inp upgrad = do
  let inplist = toList inp
      upgradlist = toList upgrad
      insize = product $ shape inp
      h = 10E-5
      finitediff i = do
        fph <- f (shift i h)
        fmh <- f (shift i (-h))
        return $ (1/(2*h)) *^ (fph - fmh)
      shift i offset = fromList (shape inp) [if j /= i then inplist!!j else inplist!!j + offset | j <- [0..insize-1]]
  listGrad <- forM [0..insize-1] $ \i -> do
    fdiff <- finitediff i
    return $ fdiff <.> upgrad
  return $ fromList (shape inp) listGrad

genericNumericBwd :: (ToTensor inp, ToTensor out, Scalar inp ~ Scalar out, Monad m, TensorDataType (Scalar inp))
                  => (inp -> m out)
                  -> inp
                  -> out
                  -> m inp
genericNumericBwd f inp upgrad = do
  let (tinp, sinp) = toTensor inp
      (tupgrad, sout) = toTensor upgrad
      tf tx = do
        let x = fromTensor tx sinp
        y <- f x
        return $ fst $ toTensor y
  tgrad <- numericBwd tf tinp tupgrad
  return $ fromTensor tgrad sinp

infixr 9 :<+
data Shape t a where
  Scalar :: Shape a a
  Tensor :: [Int] -> Shape (Tensor a) a
  Trivial :: Shape (Trivial a) a
  (:<+) :: (VectorSpace t1, VectorSpace t2, a ~ Scalar t1, a ~ Scalar t2) => Shape t1 a -> Shape t2 a -> Shape (t1, t2) a

size :: Shape t a -> Int
size Scalar = 1
size (Tensor shp) = product shp
size Trivial = 0
size (s1 :<+ s2) = size s1 + size s2

class (VectorSpace t) => ToTensor t where
  toTensorList :: t -> ([Tensor (Scalar t)], Shape t (Scalar t))

instance (TensorDataType a) => ToTensor (Tensor a) where
  toTensorList t = ([t], Tensor $ shape t)

instance ToTensor (Trivial a) where
  toTensorList Zero = ([], Trivial)

instance ToTensor CFloat where
  toTensorList x = ([fromList [1] [x]], Scalar)

instance ToTensor CDouble where
  toTensorList x = ([fromList [1] [x]], Scalar)

instance (ToTensor t1, ToTensor t2, Scalar t1 ~ Scalar t2)
         => ToTensor (t1, t2) where
  toTensorList (t1,t2) = let (lt1, s1) = toTensorList t1
                             (lt2, s2) = toTensorList t2
                         in (lt1 ++ lt2, s1 :<+ s2)

toTensor :: (ToTensor t, TensorDataType (Scalar t))
         => t -> (Tensor (Scalar t), Shape t (Scalar t))
toTensor t = let (lt, s) = toTensorList t in (concatT lt, s)

fromListShape :: (TensorDataType (Scalar t)) => [Scalar t] -> Shape t (Scalar t) -> t
fromListShape xs (Tensor shp) = fromList shp xs
fromListShape [x] Scalar = x
fromListShape [] Trivial = Zero
fromListShape xs (s1 :<+ s2) =
  let (x1, x2) = splitAt (size s1) xs
  in (fromListShape x1 s1, fromListShape x2 s2)

fromTensor :: (TensorDataType (Scalar t)) => Tensor (Scalar t) -> Shape t (Scalar t) -> t
fromTensor t s = fromListShape (toList t) s
