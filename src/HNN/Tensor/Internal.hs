{-# LANGUAGE GADTs, ForeignFunctionInterface, ScopedTypeVariables #-}
module HNN.Tensor.Internal (
  Tensor(..)
  , shape
  , nbdims
  , dtype
  , unsafeFreeze
  , unsafeThaw
  ) where
import Foreign
import Data.Proxy
import Control.Monad.Primitive

import HNN.Tensor.Mutable.Internal (MTensor(..), TensorDataType(..), fromList, toList)
import qualified Foreign.CUDA.CuDNN as CuDNN

data Tensor a where
  Tensor :: (TensorDataType a, Storable a)
         => [Int]
         -> ForeignPtr a
         -> Tensor a

shape :: Tensor a -> [Int]
shape (Tensor shp _) = shp

nbdims :: Tensor a -> Int
nbdims = length . shape

dtype :: forall a . (TensorDataType a) => Tensor a -> CuDNN.DataType
dtype _ = datatype (Proxy :: Proxy a)

-- mutable/unmutable conversions
unsafeFreeze :: (PrimMonad m) => MTensor (PrimState m) a -> m (Tensor a)
unsafeFreeze (MTensor shp ptr) = return $ Tensor shp ptr

unsafeThaw :: (PrimMonad m) => Tensor a -> m (MTensor (PrimState m) a)
unsafeThaw (Tensor shp ptr) = return $ MTensor shp ptr
