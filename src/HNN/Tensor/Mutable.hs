{-# LANGUAGE GADTs, ForeignFunctionInterface, ScopedTypeVariables #-}
module HNN.Tensor.Mutable (
  MTensor
  , IOTensor
  , TensorDataType(..)
  , Shape(..)
  , dtype
  , shaped
  , emptyTensor
  , emptyTensorP
  , withDevicePtr
  , fromList
  , toList
  , zeros
  , zerosP
  , ones
  , onesP
  , copy
  , threshInplace
  , tlog
  , texp  
  , inv
  , reshape
  , reshapeP
  ) where
import HNN.Tensor.Mutable.Internal
