module HNN.Tensor (
  Tensor
  , shape
  , nbdims
  , dtype
  , unsafeFreeze
  , unsafeThaw
  ) where
import HNN.Tensor.Internal
