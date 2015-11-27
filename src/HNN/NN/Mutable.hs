{-# LANGUAGE GADTs, ScopedTypeVariables, UndecidableInstances #-}
module HNN.NN.Mutable (
    createHandle
  , convolution2dFwd
  , convolution2dBwdFilters
  , convolution2dBwdInputs
  , ConvOutShape
  , activationFwd
  , activationBwd
  , pooling2dFwd
  , pooling2dBwd
  , PoolOutShape
  , softmaxFwd
  , softmaxBwd
  , gemmFwd
  , gemmBwdA
  , gemmBwdB
  , gemmBwdC
  , createGenerator
  , dropoutMaskIO
  , nchw_to_nhwc
  , nhwc_to_nchw
  , mul
  , addTensor
  , convolutionBackwardBias
  ) where

import Foreign
import Foreign.C
import Foreign.Marshal
import Control.Monad
import Data.Proxy
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuRAND as CuRAND
import Control.Monad.Primitive
import Unsafe.Coerce
import System.IO.Unsafe
import Control.Exception
import GHC.TypeLits

import HNN.TypeUtils
import HNN.Tensor.Mutable

-- Helper functions to deal with low-level boilerplate of CuDNN.
handleError :: String -> IO CuDNN.Status -> IO ()
handleError errMsg action = do
  status <- action
  when (status /= CuDNN.success) $ do
    errStr <- CuDNN.getErrorString status >>= peekCString
    ioError $ userError $ errStr ++ " (" ++ errMsg ++ ")"

withDescriptor :: (Storable desc)
               => String
               -> (Ptr desc -> IO CuDNN.Status) -- creation fct
               -> (desc -> IO CuDNN.Status) -- set fct
               -> (desc -> IO CuDNN.Status) -- destroy fct
               -> (desc -> IO a) -- action to perform
               -> IO a
withDescriptor name create set destroy action = do
  desc <- alloca $ \descptr -> do
    handleError ("Couldn't create " ++ name ++ " descriptor.") $ create descptr
    peek descptr
  handleError ("Couldn't set " ++ name ++ " descriptor.") $ set desc
  x <- action desc
  handleError ("Couldn't destroy " ++ name ++ " descriptor.") $ destroy desc
  return x

withTensor4d :: forall a b s . (TensorDataType a, Shape s, Nbdim s ~ 4)
             => IOTensor s a
             -> (CuDNN.TensorDescriptor -> CUDA.DevicePtr a -> IO b)
             -> IO b
withTensor4d tensor action = do
  datatype <- dtype tensor
  let [n,c,h,w] = fmap fromIntegral $ shape (Proxy :: Proxy s)
  withDescriptor
    "tensor"
    CuDNN.createTensorDescriptor
    (\d -> CuDNN.setTensor4dDescriptor d CuDNN.nchw datatype n c h w)
    CuDNN.destroyTensorDescriptor $ \tensordesc -> withDevicePtr tensor $ \dvcptr -> do
      action tensordesc dvcptr
                
withFilter4d :: forall a b s . (TensorDataType a, Shape s, Nbdim s ~ 4)
             => IOTensor s a
             -> (CuDNN.FilterDescriptor -> CUDA.DevicePtr a -> IO b)
             -> IO b
withFilter4d tensor action = do
  datatype <- dtype tensor
  let [n,c,h,w] = fmap fromIntegral $ shape (Proxy :: Proxy s)
  withDescriptor
    "filter"
    CuDNN.createFilterDescriptor
    (\d -> CuDNN.setFilter4dDescriptor d datatype n c h w)
    CuDNN.destroyFilterDescriptor $ \filtdesc -> withDevicePtr tensor $ \dvcptr -> do
      action filtdesc dvcptr

withConvDesc :: (Int,Int) -> (Int,Int) -> (Int,Int)
             -> (CuDNN.ConvolutionDescriptor -> IO a) -> IO a
withConvDesc (padh,padw) (strh,strw) (uph,upw) = do
  let [cpadh,cpadw,cstrh,cstrw,cupw,cuph] =
        map fromIntegral [padh,padw,strh,strw,uph,upw]
  withDescriptor
    "convolution"
    CuDNN.createConvolutionDescriptor
    (\d -> CuDNN.setConvolution2dDescriptor d cpadh cpadw cstrh cstrw cupw cuph CuDNN.convolution)
    CuDNN.destroyConvolutionDescriptor


-- CuDNN bindings with mutable tensors.
createHandle :: IO (CuDNN.Handle)
createHandle = alloca $ \hptr -> CuDNN.createHandle hptr >> peek hptr

type family ConvOutDim input_dim filter_dim padding stride where
  ConvOutDim input_dim filter_dim padding stride =
    1 + Quotient (input_dim + (2 * padding) - filter_dim) stride

type family ConvOutShape input_shape filter_shape padding stride where
  ConvOutShape [n1,c1,h1,w1] [n2,c1,h2,w2] [padh,padw] [strh,strw] =
    [n1,n2,ConvOutDim h1 h2 padh strh,ConvOutDim w1 w2 padw strw]

-- convolution
convolution2dFwd :: forall input_shape filter_shape padding stride out_shape m a
                 . (PrimMonad m, TensorDataType a, Shape input_shape,
                    Shape filter_shape, Shape padding, Shape stride, Shape out_shape,
                    out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                    Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4, Nbdim out_shape ~ 4)
                 => Proxy [padding,stride]
                 -> CuDNN.Handle
                 -> CuDNN.ConvolutionFwdAlgo
                 -> MTensor (PrimState m) input_shape a
                 -> MTensor (PrimState m) filter_shape a
                 -> m (MTensor (PrimState m) out_shape a)
convolution2dFwd p handle algo fmaps filters = do
  res <- unsafePrimToPrim $ convolution2dFwdIO p handle algo
         (unsafeCoerce fmaps :: IOTensor input_shape a)
         (unsafeCoerce filters :: IOTensor filter_shape a)
  return $ unsafeCoerce res

convolution2dFwdIO :: forall input_shape filter_shape padding stride out_shape a
                   . (TensorDataType a, Shape input_shape, Shape filter_shape,
                      Shape padding, Shape stride,  Shape out_shape,
                      out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                      Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4,
                      Nbdim out_shape ~ 4)
                   => Proxy [padding,stride]
                   -> CuDNN.Handle
                   -> CuDNN.ConvolutionFwdAlgo
                   -> IOTensor input_shape a
                   -> IOTensor filter_shape a
                   -> IO (IOTensor out_shape a)
convolution2dFwdIO _ handle algo fmaps filters = do
  -- Getting padding and stride into the value-level realm.
  let [padh,padw] = shape (Proxy :: Proxy padding)
      [strh,strw] = shape (Proxy :: Proxy stride)
  -- make the descriptors
  let runConv = withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
        withTensor4d fmaps $ \inputdesc inputptr -> do
          withFilter4d filters $ \filtersdesc filtersptr -> do
            output <- emptyTensor
            withTensor4d output $ \outputdesc outputptr -> do
              -- allocate workspace
              workspacesize <- alloca $ \wkspcsizeptr -> do
                handleError "Couldn't compute workspace size." $
                  CuDNN.getConvolutionForwardWorkspaceSize
                  handle inputdesc filtersdesc convdesc outputdesc algo
                  wkspcsizeptr
                peek wkspcsizeptr
              CUDA.allocaArray (fromIntegral workspacesize) $ \workspace -> do
                -- allocate alpha and beta
                withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                  -- finally run the damn thing
                  handleError "Couldn't compute convolution." $
                    CuDNN.convolutionForward
                    handle alpha inputdesc inputptr filtersdesc filtersptr
                    convdesc algo workspace workspacesize beta outputdesc
                    outputptr
            return output
  runConv `catch` \e -> do
    error $ "Exception thrown in convolution forward pass: " ++ show (e :: SomeException) ++ "\n filter shape: " ++ show (shape (Proxy :: Proxy filter_shape)) ++ ", image shape: " ++ show (shape (Proxy :: Proxy input_shape))

convolution2dBwdFilters :: forall input_shape filter_shape padding stride out_shape m a
                         . (PrimMonad m, TensorDataType a, Shape input_shape,
                            Shape filter_shape, Shape padding, Shape stride,
                            Shape out_shape,
                            out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                            Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4,
                            Nbdim out_shape ~ 4)
                         => Proxy [padding,stride]
                         -> CuDNN.Handle
                         -> MTensor (PrimState m) input_shape a
                         -> MTensor (PrimState m) filter_shape a
                         -> MTensor (PrimState m) out_shape  a
                         -> m (MTensor (PrimState m) filter_shape a)
convolution2dBwdFilters p handle fmaps filters upgrad = do
  res <- unsafePrimToPrim $ convolution2dBwdFiltersIO p handle
         (unsafeCoerce fmaps :: IOTensor input_shape a) (unsafeCoerce filters :: IOTensor filter_shape a)
         (unsafeCoerce upgrad :: IOTensor out_shape a)
  return $ unsafeCoerce res

convolution2dBwdFiltersIO :: forall input_shape filter_shape padding stride out_shape a
                          . (TensorDataType a,  Shape out_shape, Shape input_shape,
                             Shape filter_shape, Shape padding, Shape stride,
                             out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                             Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4,
                             Nbdim out_shape ~ 4)
                          => Proxy [padding,stride]
                          -> CuDNN.Handle
                          -> IOTensor input_shape a -- feature maps at which to evaluate gradient
                          -> IOTensor filter_shape a -- filters. Only shape is used.
                          -> IOTensor out_shape a -- gradient with regards to data (upper layer)
                          -> IO (IOTensor filter_shape a) -- gradients with regards to the filters
convolution2dBwdFiltersIO _ handle fmaps filters upgrad = do
  -- Getting padding and stride into the value-level realm.
  let [padh,padw] = shape (Proxy :: Proxy padding)
      [strh,strw] = shape (Proxy :: Proxy stride)
  -- make the descriptors
  let runConv = withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
        withTensor4d fmaps $ \inputdesc inputptr -> do
          withTensor4d upgrad $ \upgraddesc upgradptr -> do
            withArray [1] $ \alpha -> withArray [0] $ \beta -> do
              -- compute gradient with regards to the filters
              filtersgrad <- emptyTensor
              withFilter4d filtersgrad $ \filtersgraddesc filtersgradptr ->
                handleError
                "Couldn't compute convolution gradient with respect to filters." $
                CuDNN.convolutionBackwardFilter handle alpha inputdesc inputptr
                upgraddesc upgradptr convdesc beta filtersgraddesc filtersgradptr
              return filtersgrad
  runConv `catch` \e -> error $ "Exception thrown in convolution backward pass with regards to filters: " ++ show (e :: IOException)

convolution2dBwdInputs :: forall input_shape filter_shape padding stride out_shape m a
                       . (PrimMonad m, TensorDataType a, Shape input_shape,
                          Shape filter_shape, Shape padding, Shape stride,
                          Shape out_shape,
                          out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                          Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4,
                          Nbdim out_shape ~ 4)
                       => Proxy [padding,stride]
                       -> CuDNN.Handle
                       -> MTensor (PrimState m) input_shape a
                       -> MTensor (PrimState m) filter_shape a
                       -> MTensor (PrimState m) out_shape a
                       -> m (MTensor (PrimState m) input_shape a)
convolution2dBwdInputs p handle fmaps filters upgrad = do
  res <- unsafePrimToPrim $ convolution2dBwdInputsIO p handle
         (unsafeCoerce fmaps :: IOTensor input_shape a)
         (unsafeCoerce filters :: IOTensor filter_shape a)
         (unsafeCoerce upgrad :: IOTensor out_shape a)
  return $ unsafeCoerce res

convolution2dBwdInputsIO :: forall input_shape filter_shape padding stride out_shape a
                         . (TensorDataType a, Shape input_shape,
                            Shape filter_shape, Shape padding, Shape stride,
                            Shape out_shape,
                            out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                            Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4,
                            Nbdim out_shape ~ 4)
                         => Proxy [padding,stride]
                         -> CuDNN.Handle
                         -> IOTensor input_shape a
                         -> IOTensor filter_shape a
                         -> IOTensor out_shape a
                         -> IO (IOTensor input_shape a)
convolution2dBwdInputsIO _ handle fmaps filters upgrad = do
  -- Getting padding and stride into the value-level realm.
  let [padh,padw] = shape (Proxy :: Proxy padding)
      [strh,strw] = shape (Proxy :: Proxy stride)
  -- make the descriptors
  let runConv = withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
        withFilter4d filters $ \filtersdesc filtersptr -> do
          withTensor4d upgrad $ \upgraddesc upgradptr -> do
            withArray [1] $ \alpha -> withArray [0] $ \beta -> do
              -- compute gradient with regards to the input feature maps
              inputsgrad <- emptyTensor
              withTensor4d inputsgrad $ \inputsgraddesc inputsgradptr ->
                handleError
                "Couldn't compute convolution gradient with respect to the inputs." $
                CuDNN.convolutionBackwardData handle alpha filtersdesc filtersptr
                upgraddesc upgradptr convdesc beta inputsgraddesc inputsgradptr
              return inputsgrad
  runConv `catch` \e -> error $ "Exception thrown in convolution backward pass with regards to data: " ++ show (e :: IOException)

-- activations
activationFwd :: forall s m a . (PrimMonad m, Shape s, TensorDataType a, Nbdim s ~ 4)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) s a
              -> m (MTensor (PrimState m) s a)
activationFwd handle mode input = do
  out <- unsafePrimToPrim $
         activationFwdIO handle mode (unsafeCoerce input :: IOTensor s a)
  return $ unsafeCoerce out

activationBwd :: forall s m a . (PrimMonad m, Shape s, TensorDataType a, Nbdim s ~ 4)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) s a
              -> MTensor (PrimState m) s a
              -> MTensor (PrimState m) s a
              -> m (MTensor (PrimState m) s a)
activationBwd handle mode input output upgrad = do
  out <- unsafePrimToPrim $
         activationBwdIO handle mode (unsafeCoerce input :: IOTensor s a)
         (unsafeCoerce output :: IOTensor s a) (unsafeCoerce upgrad :: IOTensor s a)
  return $ unsafeCoerce out

activationFwdIO :: forall s a . (Shape s, TensorDataType a, Nbdim s ~ 4)
                => CuDNN.Handle
                -> CuDNN.ActivationMode
                -> IOTensor s a
                -> IO (IOTensor s a)
activationFwdIO handle mode input = do
  withTensor4d input $ \inputdesc inputptr -> do
    output <- emptyTensor
    withTensor4d output $ \outputdesc outputptr -> do
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute activations." $
          CuDNN.activationForward handle mode alpha inputdesc
          inputptr beta outputdesc outputptr
        return output

activationBwdIO :: forall s a . (Shape s, TensorDataType a, Nbdim s ~ 4)
                => CuDNN.Handle
                -> CuDNN.ActivationMode
                -> IOTensor s a
                -> IOTensor s a -- output of the forward pass on the input
                -> IOTensor s a
                -> IO (IOTensor s a)
activationBwdIO handle mode input output upgrad = do
  withTensor4d input $ \inputdesc inputptr -> do
    withTensor4d upgrad $ \upgraddesc upgradptr -> do
      grad <- emptyTensor
      withTensor4d output $ \outputdesc outputptr -> do
        withTensor4d grad $ \graddesc gradptr -> do
          withArray [1] $ \alpha -> withArray [0] $ \beta -> do
            handleError "Couldn't compute activation backward pass." $
              CuDNN.activationBackward handle mode alpha inputdesc inputptr
              upgraddesc upgradptr outputdesc outputptr beta graddesc
              gradptr
            return grad

-- 2d pooling
-- Helper to compute output shape.
pooling2dOutputShape :: (Int,Int) -> (Int,Int) -> (Int,Int) -> [Int] -> [Int]
pooling2dOutputShape (szr,szc) (padr,padc) (strr, strc) [n,ch,r,c] =
  [n,ch,outr,outc]
  where
    inr = r + padr
    inc = c + padc
    outr = (inr - overlap_r) `div` strr
    outc = (inc - overlap_c) `div` strc
    overlap_r = max 0 (szr - strr)
    overlap_c = max 0 (szc - strc)

-- Pooling output shape, at the type level.
type family PoolOutDim input_size pooling_size padding stride where
  PoolOutDim input_size pooling_size padding stride =
    Quotient (input_size + padding - Max 0 (pooling_size - stride)) stride

type family PoolOutShape input_shape pooling_shape padding stride where
  PoolOutShape [n,c,h,w] [poolh,poolw] [padh,padw] [strh,strw] =
    [n,c,PoolOutDim h poolh padh strh,PoolOutDim w poolw padw strw]

pooling2dFwd :: forall input_shape pooling_size padding stride out_shape m a
           . (PrimMonad m, TensorDataType a, Shape input_shape, Shape pooling_size,
              Shape padding, Shape stride, Shape out_shape,
              out_shape ~ PoolOutShape input_shape pooling_size padding stride,
              Nbdim input_shape ~ 4, Nbdim out_shape ~ 4)
           => Proxy [pooling_size,padding,stride]
           -> CuDNN.Handle
           -> CuDNN.PoolingMode
           -> MTensor (PrimState m) input_shape a
           -> m (MTensor (PrimState m) out_shape a)
pooling2dFwd p handle mode input = do
  out <- unsafePrimToPrim $ pooling2dFwdIO p handle mode
         (unsafeCoerce input :: IOTensor input_shape a)
  return $ unsafeCoerce out

pooling2dBwd :: forall input_shape pooling_size padding stride out_shape m a
           . (PrimMonad m, TensorDataType a, Shape input_shape, Shape pooling_size,
              Shape padding, Shape stride, Shape out_shape,
              out_shape ~ PoolOutShape input_shape pooling_size padding stride,
              Nbdim input_shape ~ 4, Nbdim out_shape ~ 4)
           => Proxy [pooling_size,padding,stride]
           -> CuDNN.Handle
           -> CuDNN.PoolingMode
           -> MTensor (PrimState m) input_shape a
           -> MTensor (PrimState m) out_shape a
           -> MTensor (PrimState m) out_shape a
           -> m (MTensor (PrimState m) input_shape a)
pooling2dBwd p handle mode inp out upgrad = do
  grad <- unsafePrimToPrim $ pooling2dBwdIO p handle mode
          (unsafeCoerce inp :: IOTensor input_shape a)
          (unsafeCoerce out :: IOTensor out_shape a)
          (unsafeCoerce upgrad :: IOTensor out_shape a)
  return $ unsafeCoerce grad

pooling2dFwdIO :: forall input_shape pooling_size padding stride out_shape a
               . (TensorDataType a, Shape input_shape, Shape pooling_size,
                  Shape padding, Shape stride, Shape out_shape,
                  out_shape ~ PoolOutShape input_shape pooling_size padding stride,
                  Nbdim input_shape ~ 4, Nbdim out_shape ~ 4)
               => Proxy [pooling_size,padding,stride]
               -> CuDNN.Handle
               -> CuDNN.PoolingMode
               -> IOTensor input_shape a
               -> IO (IOTensor out_shape a)
pooling2dFwdIO _ handle mode input = do
  -- pooling descriptor
  let [cwh,cww] = fmap fromIntegral $ shape (Proxy :: Proxy pooling_size)
      [cpadh,cpadw] = fmap fromIntegral $ shape (Proxy :: Proxy padding)
      [cstrh,cstrw] = fmap fromIntegral $ shape (Proxy :: Proxy stride)
  withDescriptor
    "pooling"
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d input $ \inputdesc inputptr -> do
        output <- emptyTensor
        -- actual pooling
        withTensor4d output $ \outputdesc outputptr -> do
          withArray [1] $ \alpha -> withArray [0] $ \beta -> do
            handleError "Couldn't compute pooling." $
              CuDNN.poolingForward handle pooldesc alpha inputdesc
              inputptr beta outputdesc outputptr
            return output

pooling2dBwdIO :: forall input_shape pooling_size padding stride out_shape a
           . (TensorDataType a, Shape input_shape, Shape pooling_size,
              Shape padding, Shape stride, Shape out_shape,
              out_shape ~ PoolOutShape input_shape pooling_size padding stride,
              Nbdim input_shape ~ 4, Nbdim out_shape ~ 4)
           => Proxy [pooling_size,padding,stride]
           -> CuDNN.Handle
           -> CuDNN.PoolingMode
           -> IOTensor input_shape a
           -> IOTensor out_shape a
           -> IOTensor out_shape a
           -> IO (IOTensor input_shape a)
pooling2dBwdIO _ handle mode inp out upgrad = do
  -- pooling descriptor
  let [cwh,cww] = fmap fromIntegral $ shape (Proxy :: Proxy pooling_size)
      [cpadh,cpadw] = fmap fromIntegral $ shape (Proxy :: Proxy padding)
      [cstrh,cstrw] = fmap fromIntegral $ shape (Proxy :: Proxy stride)
  withDescriptor
    "pooling"
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d inp $ \inpdesc inpptr -> do
        withTensor4d out $ \outdesc outptr -> do
          withTensor4d upgrad $ \upgraddesc upgradptr -> do
            grad <- emptyTensor
            withTensor4d grad $ \graddesc gradptr -> do
              withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                handleError "Couldn't compute backward pooling." $
                  CuDNN.poolingBackward handle pooldesc alpha outdesc outptr
                  upgraddesc upgradptr inpdesc inpptr beta graddesc gradptr
                return grad

-- Softmax
softmaxFwd :: forall m s a . (PrimMonad m, Shape s, Nbdim s ~ 4, TensorDataType a)
           => CuDNN.Handle
           -> CuDNN.SoftmaxAlgorithm
           -> CuDNN.SoftmaxMode
           -> MTensor (PrimState m) s a
           -> m (MTensor (PrimState m) s a)
softmaxFwd handle algo mode input = do
  res <- unsafePrimToPrim $ softmaxFwdIO handle algo mode
         (unsafeCoerce input :: IOTensor s a)
  return $ unsafeCoerce res

softmaxBwd :: forall m s a . (PrimMonad m, Shape s, Nbdim s ~ 4, TensorDataType a)
           => CuDNN.Handle
           -> CuDNN.SoftmaxAlgorithm
           -> CuDNN.SoftmaxMode
           -> MTensor (PrimState m) s a
           -> MTensor (PrimState m) s a
           -> m (MTensor (PrimState m) s a)
softmaxBwd handle algo mode src srcdiff = do
  res <- unsafePrimToPrim $ softmaxBwdIO handle algo mode
         (unsafeCoerce src :: IOTensor s a) (unsafeCoerce srcdiff :: IOTensor s a)
  return $ unsafeCoerce res

softmaxFwdIO :: (Shape s, Nbdim s ~ 4, TensorDataType a)
             => CuDNN.Handle
             -> CuDNN.SoftmaxAlgorithm
             -> CuDNN.SoftmaxMode
             -> IOTensor s a
             -> IO (IOTensor s a)
softmaxFwdIO handle algo mode input = do
  withTensor4d input $ \inpdesc inpptr -> do
    output <- emptyTensor
    withTensor4d output $ \outdesc outptr -> do
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute softmax." $
          CuDNN.softmaxForward handle algo mode alpha
          inpdesc inpptr beta outdesc outptr
        return output

softmaxBwdIO :: (Shape s, Nbdim s ~ 4, TensorDataType a)
             => CuDNN.Handle
             -> CuDNN.SoftmaxAlgorithm
             -> CuDNN.SoftmaxMode
             -> IOTensor s a
             -> IOTensor s a
             -> IO (IOTensor s a)
softmaxBwdIO handle algo mode src srcdiff = do
  withTensor4d src $ \srcdesc srcdata -> do
    withTensor4d srcdiff $ \srcdiffdesc srcdiffdata -> do
      destdiff <- emptyTensor
      withTensor4d destdiff $ \destdiffdesc destdiffdata -> do
        withArray [1] $ \alpha -> withArray [0] $ \beta -> do
          handleError "Couldn't compute softmax backward pass." $
            CuDNN.softmaxBackward handle algo mode alpha
            srcdesc srcdata srcdiffdesc srcdiffdata beta
            destdiffdesc destdiffdata
          return destdiff

-- Utility functions leveraging CuBlas.
-- GEMM wrapper, used to implement roughly everything else.
-- Computes alpha * dot(A,B) + beta * C, where all three
-- matrices are in row-major order.
-- Serves to implement nearly all the rest, including its
-- own gradients.
gemmFwd :: forall ra ca rb cb rc cc m a
        . (PrimMonad m, TensorDataType a, Shape [ra,ca], Shape [rb,cb], Shape [rc,cc])
        => Cublas.Handle
        -> Cublas.Operation
        -> Cublas.Operation
        -> a
        -> MTensor (PrimState m) [ra,ca] a
        -> MTensor (PrimState m) [rb,cb] a
        -> a
        -> MTensor (PrimState m) [rc,cc] a
        -> m ()
gemmFwd handle transa transb alpha a b beta c = do
  unsafePrimToPrim $ gemmFwdIO handle transa transb alpha
    (unsafeCoerce a :: IOTensor [ra,ca] a) (unsafeCoerce b :: IOTensor [rb,cb] b) beta
    (unsafeCoerce c :: IOTensor [rc,cc] a)

-- Composes 2 cublas operations into one.
compop :: Cublas.Operation -> Cublas.Operation -> Cublas.Operation
compop Cublas.N op = op
compop op Cublas.N = op
compop Cublas.T Cublas.T = Cublas.N
compop op1 op2 = error $ "Unsupported operations: " ++ show (op1, op2)

gemmBwdA :: forall ra ca rb cb rc cc m a
         . (PrimMonad m, TensorDataType a, Shape [ra,ca], Shape [rb,cb], Shape [rc,cc])
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) [rb,cb] a
         -> MTensor (PrimState m) [rc,cc] a
         -> MTensor (PrimState m) [ra,ca] a
         -> m ()
gemmBwdA handle transa transb alpha b upgrad out = do
  -- Need to distinguish between the case where
  -- we have an operation on A or not.
  case transa of
    Cublas.N -> gemmFwd handle Cublas.N (compop transb Cublas.T)
                alpha upgrad b 0 out
    Cublas.T -> gemmFwd handle transb Cublas.T alpha b upgrad 0 out

gemmBwdB :: forall ra ca rb cb rc cc m a
         . (PrimMonad m, TensorDataType a, Shape [ra,ca], Shape [rb,cb], Shape [rc,cc])
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) [ra,ca] a
         -> MTensor (PrimState m) [rc,cc] a
         -> MTensor (PrimState m) [rb,cb] a
         -> m ()
gemmBwdB handle transa transb alpha a upgrad out = do
  case transb of
    Cublas.N -> gemmFwd handle (compop transa Cublas.T) Cublas.N alpha
                a upgrad 0 out
    Cublas.T -> gemmFwd handle Cublas.T transa alpha upgrad a 0 out

gemmBwdC :: forall rc cc m a
         . (PrimMonad m, TensorDataType a, Shape [rc,cc])
         => Cublas.Handle
         -> a
         -> MTensor (PrimState m) [rc,cc] a
         -> MTensor (PrimState m) [rc,cc] a
         -> m ()
gemmBwdC handle beta upgrad out = do
  unsafePrimToPrim $ gemmBwdCIO handle beta
    (unsafeCoerce upgrad :: IOTensor [rc,cc] a) (unsafeCoerce out :: IOTensor [rc,cc] a)
  
gemmBwdCIO :: forall rc cc a
           . (TensorDataType a, Shape [rc,cc])
           => Cublas.Handle
           -> a
           -> IOTensor [rc,cc] a
           -> IOTensor [rc,cc] a
           -> IO ()
gemmBwdCIO handle beta upgrad out = do
  withDevicePtr upgrad $ \upgradptr -> do
    withDevicePtr out $ \outptr -> do
      Cublas.copy handle (size (Proxy :: Proxy [rc,cc])) upgradptr 1 outptr 1
      Cublas.scal handle (size (Proxy :: Proxy [rc,cc])) beta outptr 1
  
gemmFwdIO :: forall ra ca rb cb rc cc a
          . (TensorDataType a, Shape [ra,ca], Shape [rb,cb], Shape [rc,cc])
          => Cublas.Handle
          -> Cublas.Operation
          -> Cublas.Operation
          -> a
          -> IOTensor [ra,ca] a
          -> IOTensor [rb,cb] a
          -> a
          -> IOTensor [rc,cc] a
          -> IO ()
gemmFwdIO handle transa transb alpha a b beta c = do
  -- Figuring out the parameters to pass GEMM so it does
  -- the right thing in row-major layout, depending on the
  -- user requested transforms.
  let [ra, ca] = shape (Proxy :: Proxy [ra,ca])
      [rb, cb] = shape (Proxy :: Proxy [rb,cb])
      [rc, cc] = shape (Proxy :: Proxy [rc,cc])
  -- The rules for m, n and k were derived on paper
  -- by considering all 4 possible cases, taking into
  -- account that due to layout differences gemm will
  -- read all matrices as transpose, in addition to
  -- the user supplied transformation. This also relies
  -- on the property that the transpose operation
  -- will work on both column-major and row-major data
  -- seamlessly.
  let shapeError = error $
        "Incompatible shapes/operations combination for matrix multiplication: "
        ++ show [ra, ca] ++ ", " ++ show [rb, cb] ++ ", " ++ show [rc, cc]
        ++ ", " ++ show transa ++ ", " ++ show transb
  (m, n, k) <- case (transa, transb) of
    (Cublas.N, Cublas.N) -> do
      when (ca /= rb) shapeError
      return (cb, ra, rb)
    (Cublas.T, Cublas.N) -> do
      when (ra /= rb) shapeError
      return (cb, ca, ra)
    (Cublas.N, Cublas.T) -> do
      when (ca /= cb) shapeError
      return (rb, ra, ca)
    (Cublas.T, Cublas.T) -> do
      when (ra /= cb) shapeError
      return (rb, ca, ra)
  -- since row-major matrices will be read as transposed,
  -- the leading dimension are their number of columns.
  let
    lda = ca
    ldb = cb
    ldc = cc
  withDevicePtr a $ \aptr -> do
    withDevicePtr b $ \bptr -> do
      withDevicePtr c $ \cptr -> do
        Cublas.gemm handle transb transa m n k alpha bptr ldb aptr lda beta cptr ldc

-- dropout
createGenerator :: CuRAND.RngType -> IO (CuRAND.Generator)
createGenerator rngtype = do
  alloca $ \genptr -> do
    CuRAND.createGenerator genptr rngtype
    peek genptr

-- compute a random mask of ones and zeros
-- to apply elementwise to the input tensor.
dropoutMaskIO :: forall a s . (TensorDataType a, Shape s)
              => CuRAND.Generator
              -> a
              -> IO (IOTensor s a)
dropoutMaskIO gen drop_proba = do
  -- Simple algo for dropout of activations:
  -- * generate an array of random values between 0 and 1
  -- * threshold that array with the dropout probability
  -- * elementwise multiply the input array with it
  rand_array <- emptyTensor
  withDevicePtr rand_array $ \randarrayptr -> do
    -- generate random array
    generateUniform gen randarrayptr $ fromIntegral $ size (Proxy :: Proxy s)
    -- threshold it
    thresh randarrayptr (fromIntegral $ size (Proxy :: Proxy s)) drop_proba randarrayptr
    return rand_array

-- element-wise multiply (in place)
mul :: forall a m s . (TensorDataType a, PrimMonad m, Shape s)
    => MTensor (PrimState m) s a
    -> MTensor (PrimState m) s a
    -> m ()
mul t1 t2 =
  unsafePrimToPrim $ mulIO
  (unsafeCoerce t1 :: IOTensor s a) (unsafeCoerce t2 :: IOTensor s a)

mulIO :: forall a s . (TensorDataType a, Shape s)
      => IOTensor s a
      -> IOTensor s a
      -> IO ()
mulIO t1 t2 = do
  withDevicePtr t1 $ \t1ptr ->
    withDevicePtr t2 $ \t2ptr -> do
      rawMul t1ptr t2ptr $ fromIntegral $ size (Proxy :: Proxy s)

nchw_to_nhwc :: forall n c h w a m
             . (PrimMonad m, TensorDataType a, KnownNat n, KnownNat c, KnownNat h,
                KnownNat w)
             => CuDNN.Handle
             -> MTensor (PrimState m) [n,c,h,w] a
             -> m (MTensor (PrimState m) [n,h,w,c] a)
nchw_to_nhwc handle src = unsafePrimToPrim $ do
  res <- transformTensorIO handle CuDNN.nchw CuDNN.nhwc
         (unsafeCoerce src :: IOTensor [n,c,h,w] a) :: IO (IOTensor [n,h,w,c] a)
  return $ unsafeCoerce res

nhwc_to_nchw :: forall n c h w a m
             . (PrimMonad m, TensorDataType a, KnownNat n, KnownNat c, KnownNat h,
                KnownNat w)
             => CuDNN.Handle
             -> MTensor (PrimState m) [n,h,w,c] a
             -> m (MTensor (PrimState m) [n,c,h,w] a)
nhwc_to_nchw handle src = unsafePrimToPrim $ do
  res <- transformTensorIO handle CuDNN.nhwc CuDNN.nchw
         (unsafeCoerce src :: IOTensor [n,h,w,c] a) :: IO (IOTensor [n,c,h,w] a)
  return $ unsafeCoerce res

-- Unsafe but useful.
transformTensorIO :: forall a s1 s2
                  . (TensorDataType a, Shape s1, Shape s2,
                     Nbdim s1 ~ 4, Nbdim s2 ~ 4)
                  => CuDNN.Handle
                  -> CuDNN.TensorFormat
                  -> CuDNN.TensorFormat
                  -> IOTensor s1 a
                  -> IO (IOTensor s2 a)
transformTensorIO handle srcf dstf src = do
  datatype <- dtype src
  let [a',b',c',d'] = shape (Proxy :: Proxy s1)
      [n,c,h,w] = if srcf == CuDNN.nchw then [a',b',c',d']
                  else if srcf == CuDNN.nhwc then [a',d',b',c']
                       else error $ "Unsupported input format: " ++ show srcf
      [cn,cc,ch,cw] = fmap fromIntegral [n,c,h,w]
  withDescriptor
    "tensor"
    CuDNN.createTensorDescriptor
    (\d -> CuDNN.setTensor4dDescriptor d srcf datatype cn cc ch cw)
    CuDNN.destroyTensorDescriptor $ \srcdesc -> do
      withDescriptor
        "tensor"
        CuDNN.createTensorDescriptor
        (\d -> CuDNN.setTensor4dDescriptor d dstf datatype cn cc ch cw)
        CuDNN.destroyTensorDescriptor $ \dstdesc -> do
          dst <- emptyTensor
          withDevicePtr src $ \srcptr -> do
            withDevicePtr dst $ \dstptr -> do
              withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                handleError "Couldn't transform the tensor." $
                  CuDNN.transformTensor handle alpha srcdesc srcptr beta
                  dstdesc dstptr
                return dst

addTensorIO :: (TensorDataType a, Shape s, Nbdim s ~ 4)
            => CuDNN.Handle
            -> CuDNN.AddMode
            -> IOTensor s a
            -> IOTensor s a
            -> IO (IOTensor s a)
addTensorIO handle addMode bias src = do
  out <- copy src
  withTensor4d bias $ \biasdesc biasptr -> do
    withTensor4d out $ \outdesc outptr -> do
      withArray [1] $ \alpha -> withArray [1] $ \beta -> do
        handleError "Couldn't add the tensors." $
          CuDNN.addTensor handle addMode alpha biasdesc biasptr
          beta outdesc outptr
  return out

addTensor :: forall a m s . (TensorDataType a, PrimMonad m, Shape s, Nbdim s ~ 4)
          => CuDNN.Handle
          -> CuDNN.AddMode
          -> MTensor (PrimState m) s a
          -> MTensor (PrimState m) s a
          -> m (MTensor (PrimState m) s a)
addTensor handle addMode bias src = do
  out <- unsafePrimToPrim $ addTensorIO handle addMode
         (unsafeCoerce bias :: IOTensor s a) (unsafeCoerce src :: IOTensor s a)
  return $ unsafeCoerce out

convolutionBackwardBiasIO :: (TensorDataType a, KnownNat n, KnownNat c, KnownNat h,
                              KnownNat w)
                          => CuDNN.Handle
                          -> IOTensor [n,c,h,w] a
                          -> IO (IOTensor [1,c,1,1] a)
convolutionBackwardBiasIO handle upgrad = do
  biasgrad <- emptyTensor
  withTensor4d upgrad $ \upgraddesc upgradptr -> do
    withTensor4d biasgrad $ \biasgraddesc biasgradptr -> do
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute convolution backward pass with regards to bias" $
          CuDNN.convolutionBackwardBias handle alpha upgraddesc upgradptr
          beta biasgraddesc biasgradptr
  return biasgrad

convolutionBackwardBias :: forall n c h w a m
                        . (TensorDataType a, PrimMonad m, KnownNat n, KnownNat c,
                           KnownNat h, KnownNat w)
                        => CuDNN.Handle
                        -> MTensor (PrimState m) [n,c,h,w] a
                        -> m (MTensor (PrimState m) [1,c,1,1] a)
convolutionBackwardBias handle upgrad = do
  biasgrad <- unsafePrimToPrim $ convolutionBackwardBiasIO handle
              (unsafeCoerce upgrad :: IOTensor [n,c,h,w] a)
  return $ unsafeCoerce biasgrad
