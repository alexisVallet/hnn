{-# LANGUAGE ForeignFunctionInterface #-}
{-|
FFI wrapper to custom CUDA code needed for for HNN.
-}
module HNN.Internal.Cubits where

import Foreign
import Foreign.C
import Foreign.CUDA.Types

#include <hnn_cubits.h>

foreign import ccall  "thresh"
  thresh :: DevicePtr CFloat -> CSize -> CFloat -> DevicePtr CFloat
         -> IO ()

foreign import ccall  "threshDouble"
  threshDouble :: DevicePtr CDouble -> CSize -> CDouble
               -> DevicePtr CDouble -> IO ()

foreign import ccall  "mul"
  mul :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> IO ()

foreign import ccall  "mulDouble"
  mulDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> IO ()

foreign import ccall  "add"
  add :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> IO ()

foreign import ccall  "addDouble"
  addDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> IO ()

foreign import ccall  "tabs"
  tabs :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall  "tabsDouble"
  tabsDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall  "signum"
  tsignum :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall  "signumDouble"
  tsignumDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall  "subtract"
  subtract :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> IO ()

foreign import ccall  "subtractDouble"
  subtractDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> IO ()

foreign import ccall  "negate"
  tnegate :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall  "negateDouble"
  tnegateDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall  "scale"
  scale :: CFloat -> DevicePtr CFloat -> CSize -> IO ()

foreign import ccall  "scaleDouble"
  scaleDouble :: CDouble -> DevicePtr CDouble -> CSize -> IO ()

foreign import ccall  "logFloat"
  logFloat :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall  "logDouble"
  logDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall  "inv"
  inv :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall  "invDouble"
  invDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "texp"
  texp :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "texpDouble"
  texpDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tsqrt"
  tsqrt :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tsqrtDouble"
  tsqrtDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tsin"
  tsin :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tsinDouble"
  tsinDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tcos"
  tcos :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tcosDouble"
  tcosDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "ttan"
  ttan :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "ttanDouble"
  ttanDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tasin"
  tasin :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tasinDouble"
  tasinDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tacos"
  tacos :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tacosDouble"
  tacosDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tatan"
  tatan :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tatanDouble"
  tatanDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tsinh"
  tsinh :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tsinhDouble"
  tsinhDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tcosh"
  tcosh :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tcoshDouble"
  tcoshDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "ttanh"
  ttanh :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "ttanhDouble"
  ttanhDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tasinh"
  tasinh :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tasinhDouble"
  tasinhDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tacosh"
  tacosh :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tacoshDouble"
  tacoshDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tatanh"
  tatanh :: DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tatanhDouble"
  tatanhDouble :: DevicePtr CDouble -> CSize -> IO ()

foreign import ccall "tpow"
  tpow :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> IO ()

foreign import ccall "tpowDouble"
  tpowDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> IO ()

