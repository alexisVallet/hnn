module HNN.Tensor (
  Tensor
  , TensorDataType
  , Shape(..)
  , dtype
  , reshape
  , unsafeFreeze
  , unsafeThaw
  , fromList
  , toList
  , fromVector
  , toVector
  , tconcat
  , tsplit
  , elementwiseMax
  , module Data.VectorSpace
  ) where
import Data.VectorSpace
import HNN.Tensor.Internal
