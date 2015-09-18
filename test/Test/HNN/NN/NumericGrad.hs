{-# LANGUAGE ScopedTypeVariables #-}
module Test.HNN.NN.NumericGrad where

import Numeric.LinearAlgebra.HMatrix
import Control.Monad

-- Computes the numerical backward pass of
-- an arbitrary pure vector computation.
numericBwd :: (Vector Float -> Vector Float) -- computation
           -> Vector Float -- point at which to evaluate gradient
           -> Vector Float -- gradient from upper layer of computation
           -> Vector Float -- gradient for the computation
numericBwd f x y =
  fromList [dot (finitediff i) y | i <- [1..size x]]
  where finitediff i = (f (shift i h) - f (shift i (-h))) / scalar (2 * h)
        h = 10E-5
        shift i offset = fromList [x!i + if j == i then offset else 0
                                   | j <- [1..size x]]

monadicNumericBwd :: Monad m
                  => (Vector Float -> m (Vector Float))
                  -> Vector Float
                  -> Vector Float
                  -> m (Vector Float)
monadicNumericBwd f x y = do
  let h = 10E-5
      shift i offset = fromList [x!i + if j == i then offset else 0
                                 | j <- [1..size x]]
  grad <- forM [1..size x] $ \i -> do
    f1res <- f (shift i h)
    f2res <- f (shift i (-h))
    return $ dot ((f1res - f2res) / scalar (2 * h)) y
  return $ fromList grad
