module Test.HNN.NN.NumericGrad where

import Numeric.LinearAlgebra.HMatrix

-- Computes the numerical backward pass of
-- an arbitrary vector computation.
numericBwd :: Numeric a
           => (Vector a -> Vector a) -- computation
           -> Vector a -- point at which to evaluate gradient
           -> Vector a -- gradient from upper layer of computation
           -> Vector a -- gradient for the computation
numericBwd f x y =
  fromList [dot (finitediff i) y | i <- [1..size x]]
  where finitediff i = (f (shift i h) - f (shift i -h)) / (2*h)
        h = 10E-5
        shift i offset = fromList [x!i + if j == i then offset else 0
                                   | j <- [1..size x]]
