{-# LANGUAGE UndecidableInstances #-}
{-| Utilities for type-level computations -}
module HNN.TypeUtils (
    Quotient
  , Remainder
  , Max
  ) where

import GHC.TypeLits

-- Type-level Euclidean division and remainder for naturals.
type family Quotient' (m :: Nat) (n :: Nat) (ord :: Ordering) where
  Quotient' m n 'LT = 0
  Quotient' m n 'EQ = 1
  Quotient' m n 'GT = 1 + Quotient' (m - n) n (CmpNat (m - n) n)

type family Remainder' (m :: Nat) (n :: Nat) (ord :: Ordering) where
  Remainder' m n 'LT = m
  Remainder' m n 'EQ = 0
  Remainder' m n 'GT = Remainder' (m - n) n (CmpNat (m - n) n)

type family Quotient (m :: Nat) (n :: Nat) where
  Quotient m n = Quotient' m n (CmpNat m n)

type family Remainder (m :: Nat) (n :: Nat) where
  Remainder m n = Remainder' m n (CmpNat m n)

-- Type level max for naturals.
type family Max' (m :: Nat) (n :: Nat) (ord :: Ordering) where
  Max' m n 'LT = n
  Max' m n o = m

type family Max (m :: Nat) (n :: Nat) where
  Max m n = Max' m n (CmpNat m n)
