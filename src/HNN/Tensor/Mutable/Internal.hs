{-# LANGUAGE UndecidableInstances #-}
module HNN.Tensor.Mutable.Internal (
  MTensor(..)
  , IOTensor(..)
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
import Foreign
import Foreign.C
import Foreign.Marshal
import Foreign.Concurrent
import System.IO.Unsafe
import Data.List
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuRAND as CuRAND
import Data.Proxy
import System.IO.Error
import Control.Monad
import Control.Monad.Primitive
import Unsafe.Coerce
import Data.VectorSpace
import GHC.Generics
import GHC.TypeLits
import Data.Serialize
import Control.DeepSeq

import qualified HNN.Internal.Cubits as Cubits

instance Generic CFloat where
  type Rep CFloat = Rep Float
  from cx = from (realToFrac cx :: Float)
  to rep = realToFrac (to rep :: Float)

instance Generic CDouble where
  type Rep CDouble = Rep Double
  from cx = from (realToFrac cx :: Double)
  to rep = realToFrac (to rep :: Double)

instance Serialize CFloat
instance Serialize CDouble
instance NFData CFloat
instance NFData CDouble

class (Cublas.Cublas a, Floating a, Storable a, VectorSpace a, a ~ Scalar a, Serialize a, NFData a)
      => TensorDataType a where
  datatype :: Proxy a -> CuDNN.DataType
  -- ad-hoc stuff to cleanly wrap low-level C APIs
  thresh :: CUDA.DevicePtr a -> CSize -> a -> CUDA.DevicePtr a -> IO ()
  rawMul :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawAdd :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawAbs :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSignum :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSubtract :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawNegate :: CUDA.DevicePtr a -> CSize -> IO ()
  rawScale :: a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawLog :: CUDA.DevicePtr a -> CSize -> IO ()
  rawInv :: CUDA.DevicePtr a -> CSize -> IO ()
  rawExp :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSqrt :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSin :: CUDA.DevicePtr a -> CSize -> IO ()
  rawCos :: CUDA.DevicePtr a -> CSize -> IO ()
  rawTan :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAsin :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAcos :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAtan :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSinh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawCosh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawTanh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAsinh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAcosh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAtanh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawPow :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawMax :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  -- curand stuff
  generateUniform :: CuRAND.Generator
                  -> CUDA.DevicePtr a
                  -> CSize
                  -> IO CuRAND.Status
  generateNormal :: CuRAND.Generator
                 -> CUDA.DevicePtr a
                 -> CSize
                 -> a
                 -> a
                 -> IO CuRAND.Status
  generateLogNormal :: CuRAND.Generator
                    -> CUDA.DevicePtr a
                    -> CSize
                    -> a
                    -> a
                    -> IO CuRAND.Status

instance TensorDataType CFloat where
  datatype = const CuDNN.float
  thresh = Cubits.thresh
  rawMul = Cubits.mul
  rawAdd = Cubits.add
  rawAbs = Cubits.tabs
  rawSignum = Cubits.tsignum
  rawSubtract = Cubits.subtract
  rawNegate = Cubits.tnegate
  rawScale = Cubits.scale
  rawLog = Cubits.logFloat
  rawInv = Cubits.inv
  rawExp = Cubits.texp
  rawSqrt = Cubits.tsqrt
  rawSin = Cubits.tsin
  rawCos = Cubits.tcos
  rawTan = Cubits.ttan
  rawAsin = Cubits.tasin
  rawAcos = Cubits.tacos
  rawAtan = Cubits.tatan
  rawSinh = Cubits.tsinh
  rawCosh = Cubits.tcosh
  rawTanh = Cubits.ttanh
  rawAsinh = Cubits.tasinh
  rawAcosh = Cubits.tacosh
  rawAtanh = Cubits.tatanh
  rawPow = Cubits.tpow
  rawMax = Cubits.tmax
  generateUniform = CuRAND.generateUniform
  generateNormal = CuRAND.generateNormal
  generateLogNormal = CuRAND.generateLogNormal

instance TensorDataType CDouble where
  datatype = const CuDNN.double
  thresh = Cubits.threshDouble
  rawMul = Cubits.mulDouble
  rawAdd = Cubits.addDouble
  rawAbs = Cubits.tabsDouble
  rawSignum = Cubits.tsignumDouble
  rawSubtract = Cubits.subtractDouble
  rawNegate = Cubits.tnegateDouble
  rawScale = Cubits.scaleDouble
  rawLog = Cubits.logDouble
  rawInv = Cubits.invDouble
  rawExp = Cubits.texpDouble
  rawSqrt = Cubits.tsqrtDouble
  rawSin = Cubits.tsinDouble
  rawCos = Cubits.tcosDouble
  rawTan = Cubits.ttanDouble
  rawAsin = Cubits.tasinDouble
  rawAcos = Cubits.tacosDouble
  rawAtan = Cubits.tatanDouble
  rawSinh = Cubits.tsinhDouble
  rawCosh = Cubits.tcoshDouble
  rawTanh = Cubits.ttanhDouble
  rawAsinh = Cubits.tasinhDouble
  rawAcosh = Cubits.tacoshDouble
  rawAtanh = Cubits.tatanhDouble
  rawPow = Cubits.tpowDouble
  rawMax = Cubits.tmaxDouble
  generateUniform = CuRAND.generateUniformDouble
  generateNormal = CuRAND.generateNormalDouble
  generateLogNormal = CuRAND.generateLogNormalDouble

-- mutable tensor with type-level fixed shape.
data MTensor st (shp :: [Nat]) a = MTensor (ForeignPtr a)

type IOTensor = MTensor RealWorld

class Shape (s :: [Nat]) where
  type Size s :: Nat
  type Nbdim s :: Nat
  shape :: Proxy s -> [Int]
  nbdim :: Proxy s -> Int
  nbdim p = length $ shape p
  size :: Proxy s -> Int
  size p = product $ shape p

instance Shape '[] where
  type Size '[] = 0
  type Nbdim '[] = 0
  shape _ = []

instance (KnownNat n) => Shape '[n] where
  type Size '[n] = n
  type Nbdim '[n] = 1
  shape _ = [fromIntegral $ natVal (Proxy :: Proxy n)]

instance forall (e1 :: Nat) (e2 :: Nat) (l :: [Nat])
         . (KnownNat e1, Shape (e2 ': l))
         => Shape (e1 ': (e2 ': l)) where
  type Size (e1 ': (e2 ': l)) = e1 * Size (e2 ': l)
  type Nbdim (e1 ': (e2 ': l)) = 1 + Nbdim (e2 ': l)
  shape _ = (fromIntegral $ natVal (Proxy :: Proxy e1)) : shape (Proxy :: Proxy (e2 ': l))

dtype :: forall m s a . (PrimMonad m, TensorDataType a)
      => MTensor (PrimState m) s a
      -> m CuDNN.DataType
dtype _ = return $ datatype (Proxy :: Proxy a)

shaped :: Proxy s -> m (MTensor st s a) -> m (MTensor st s a)
shaped _ mt = mt

emptyTensor :: forall m s a . (TensorDataType a, Shape s, PrimMonad m)
            => m (MTensor (PrimState m) s a)
emptyTensor = do
  res <- unsafePrimToPrim $ (emptyTensorIO :: IO (IOTensor s a))
  return $ unsafeCoerce res

emptyTensorP :: forall m s a . (TensorDataType a, Shape s, PrimMonad m)
            => Proxy s -> m (MTensor (PrimState m) s a)
emptyTensorP _ = do
  res <- unsafePrimToPrim $ (emptyTensorIO :: IO (IOTensor s a))
  return $ unsafeCoerce res

emptyTensorIO :: forall s a . (TensorDataType a, Shape s) => IO (IOTensor s a)
emptyTensorIO = do
  dvcptr <- CUDA.mallocArray $ size (Proxy :: Proxy s)
  let finalizer = CUDA.free dvcptr
  datafptr <- Foreign.Concurrent.newForeignPtr
              (CUDA.useDevicePtr dvcptr)
              finalizer
  return $ MTensor datafptr

withDevicePtr :: (Storable a) => IOTensor s a -> (CUDA.DevicePtr a -> IO b) -> IO b
withDevicePtr (MTensor datafptr) action = do
  withForeignPtr datafptr $ \dataptr -> action (CUDA.DevicePtr dataptr)

makeTensor :: forall s a . (TensorDataType a, Shape s)
           => Ptr a
           -> IO (IOTensor s a)
makeTensor dataptr = do
  tensor <- emptyTensor
  withDevicePtr tensor $ \dvcptr ->
    CUDA.pokeArray (size (Proxy :: Proxy s)) dataptr dvcptr
  return tensor

fromList :: forall m s a . (PrimMonad m, TensorDataType a, Shape s)
         =>  [a] -> m (MTensor (PrimState m) s a)
fromList content = do
  res <- unsafePrimToPrim $ (fromListIO content :: IO (IOTensor s a))
  return $ unsafeCoerce res

fromListIO :: forall s a . (TensorDataType a, Shape s)
           => [a]
           -> IO (IOTensor s a)
fromListIO datalist = do
  when (size (Proxy :: Proxy s) /= length datalist) $ ioError $ userError
    "Shape is incompatible with provided data."
  withArray datalist $ \dataptr ->
    makeTensor dataptr

zeros :: forall m s a . (PrimMonad m, Shape s, TensorDataType a)
      => m (MTensor (PrimState m) s a)
zeros = fromList $ take (size (Proxy :: Proxy s)) $ repeat 0

ones :: forall m s a . (PrimMonad m, Shape s, TensorDataType a)
      => m (MTensor (PrimState m) s a)
ones = fromList $ take (size (Proxy :: Proxy s)) $ repeat 1

zerosP :: forall m s a . (PrimMonad m, Shape s, TensorDataType a)
       => Proxy s -> m (MTensor (PrimState m) s a)
zerosP p = fromList $ take (size p) $ repeat 0

onesP :: forall m s a . (PrimMonad m, Shape s, TensorDataType a)
      => Proxy s -> m (MTensor (PrimState m) s a)
onesP p = fromList $ take (size p) $ repeat 1


toList :: forall m s a . (PrimMonad m, Shape s, TensorDataType a)
       => MTensor (PrimState m) s a -> m [a]
toList tensor = unsafePrimToPrim $ toListIO (unsafeCoerce tensor :: IOTensor s a)

toListIO :: forall s a . (Shape s, TensorDataType a) => IOTensor s a -> IO [a]
toListIO tensor = do
  withDevicePtr tensor (CUDA.peekListArray (size (Proxy :: Proxy s)))

copy :: forall m s a . (PrimMonad m, Shape s, TensorDataType a)
     => MTensor (PrimState m) s a -> m (MTensor (PrimState m) s a)
copy tensor = do
  res <- unsafePrimToPrim $ copyIO (unsafeCoerce tensor :: IOTensor s a)
  return $ unsafeCoerce res

copyIO :: forall s a . (Shape s, TensorDataType a) => IOTensor s a -> IO (IOTensor s a)
copyIO tensor = do
  out <- emptyTensorIO
  withDevicePtr tensor $ \tensorptr -> do
    withDevicePtr out $ \outptr -> do
      CUDA.copyArray (size (Proxy :: Proxy s)) tensorptr outptr
      return out

threshInplace :: forall m s . (PrimMonad m, Shape s)
              => MTensor (PrimState m) s CFloat -> CFloat -> m ()
threshInplace tensor threshold =
  unsafePrimToPrim $ threshInplaceIO (unsafeCoerce tensor :: IOTensor s CFloat) threshold

threshInplaceIO :: forall s . (Shape s) => IOTensor s CFloat -> CFloat -> IO ()
threshInplaceIO tensor threshold = do
  withDevicePtr tensor $ \tensorptr -> do
    thresh tensorptr (fromIntegral $ size (Proxy :: Proxy s)) threshold tensorptr

tlog :: forall m s a . (PrimMonad m, Shape s, TensorDataType a)
     => MTensor (PrimState m) s a -> m ()
tlog tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor s a
  withDevicePtr iotensor $ \tptr -> do
    rawLog tptr (fromIntegral $ size (Proxy :: Proxy s))

texp :: forall m s a . (PrimMonad m, Shape s, TensorDataType a)
     => MTensor (PrimState m) s a -> m ()
texp t = unsafePrimToPrim $ do
  let iot = unsafeCoerce t :: IOTensor s a
  withDevicePtr iot $ \tptr -> do
    rawExp tptr (fromIntegral $ size (Proxy :: Proxy s))

inv :: forall m s a . (PrimMonad m, Shape s, TensorDataType a)
    => MTensor (PrimState m) s a -> m ()
inv tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor s a
  withDevicePtr iotensor $ \tptr -> do
    rawInv tptr $ fromIntegral $ size (Proxy :: Proxy s)

reshape :: (Shape s1, Shape s2, Size s1 ~ Size s2)
        => MTensor st s1 a -> MTensor st s2 a
reshape = unsafeCoerce

reshapeP :: (Shape s1, Shape s2, Size s1 ~ Size s2)
         => Proxy s2 -> MTensor st s1 a -> MTensor st s2 a
reshapeP _ = unsafeCoerce
