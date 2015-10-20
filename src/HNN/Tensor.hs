module HNN.Tensor (
  Tensor
  , TensorDataType
  , shape
  , nbdims
  , dtype
  , reshape
  , unsafeFreeze
  , unsafeThaw
  , fromList
  , toList
  , fromVector
  , toVector
  , concatT
  , module Data.VectorSpace
  ) where
import Data.VectorSpace
import HNN.Tensor.Internal
