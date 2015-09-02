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

foreign import ccall unsafe "add"
  add :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> IO ()

foreign import ccall unsafe "addDouble"
  addDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> IO ()

foreign import ccall unsafe "tabs"
  tabs :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall unsafe "tabsDouble"
  tabsDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall unsafe "signum"
  tsignum :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall unsafe "signumDouble"
  tsignumDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall unsafe "subtract"
  subtract :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> IO ()

foreign import ccall unsafe "subtractDouble"
  subtractDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> IO ()
