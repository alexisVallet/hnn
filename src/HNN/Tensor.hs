module HNN.Tensor (
  Tensor
  , TensorDataType(..)
  , shape
  , nbdims
  , dtype
  , unsafeFreeze
  , unsafeThaw
  ) where
import HNN.Tensor.Internal
