{-# LANGUAGE GADTs, ForeignFunctionInterface, ScopedTypeVariables #-}
module HNN.Tensor.Mutable (
  MTensor
  , IOTensor
  , TensorDataType(..)
  , nbdims
  , dtype
  , shape
  , emptyTensor
  , withDevicePtr
  , fromList
  , toList
  ) where
import HNN.Tensor.Mutable.Internal
