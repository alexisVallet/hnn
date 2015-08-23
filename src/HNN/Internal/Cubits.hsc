{-# LANGUAGE ForeignFunctionInterface #-}
{-|
FFI wrapper to custom CUDA code needed for for HNN.
-}
module HNN.Internal.Cubits where

import Foreign
import Foreign.C
import Foreign.CUDA.Types

#include <hnn_cubits.h>

foreign import ccall unsafe "thresh"
  thresh :: DevicePtr CFloat -> CSize -> CFloat -> DevicePtr CFloat
         -> IO ()

foreign import ccall unsafe "threshDouble"
  threshDouble :: DevicePtr CDouble -> CSize -> CDouble
               -> DevicePtr CDouble -> IO ()

foreign import ccall unsafe "mul"
  mul :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> IO ()

foreign import ccall unsafe "mulDouble"
  mulDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> IO ()

