{-# LANGUAGE GADTs, ScopedTypeVariables #-}
module HNN.NN.Mutable (
  createHandle
  , convolution2dFwd
  , convolution2dBwdFilters
  , convolution2dBwdInputs
  , activationFwd
  , activationBwd
  , pooling2dFwd
  , pooling2dBwd
  ) where

import Foreign
import Foreign.C
import Foreign.Marshal
import Control.Monad
import Data.Proxy
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as Cublas
import Control.Monad.Primitive
import Unsafe.Coerce

import HNN.Tensor.Mutable

-- Helper functions to deal with low-level boilerplate of CuDNN.
handleError :: String -> IO CuDNN.Status -> IO ()
handleError errMsg action = do
  status <- action
  when (status /= CuDNN.success) $ do
    errStr <- CuDNN.getErrorString status >>= peekCString
    ioError $ userError $ errStr ++ " (" ++ errMsg ++ ")"

withDescriptor :: (Storable desc)
               => (Ptr desc -> IO CuDNN.Status) -- creation fct
               -> (desc -> IO CuDNN.Status) -- set fct
               -> (desc -> IO CuDNN.Status) -- destroy fct
               -> (desc -> IO a) -- action to perform
               -> IO a
withDescriptor create set destroy action = do
  desc <- alloca $ \descptr -> do
    handleError "Couldn't create descriptor." $ create descptr
    peek descptr
  handleError "Couldn't set descriptor." $ set desc
  x <- action desc
  handleError "Couldn't destroy descriptor." $ destroy desc
  return x

withTensor4d :: (TensorDataType a)
             => IOTensor a
             -> (CuDNN.TensorDescriptor -> CUDA.DevicePtr a -> IO b)
             -> IO b
withTensor4d tensor action = do
  datatype <- dtype tensor
  [n,c,h,w] <- shape tensor >>= return . map fromIntegral
  withDescriptor
    CuDNN.createTensorDescriptor
    (\d -> CuDNN.setTensor4dDescriptor d CuDNN.nchw datatype n c h w)
    CuDNN.destroyTensorDescriptor $ \tensordesc -> withDevicePtr tensor $ \dvcptr -> do
      action tensordesc dvcptr
      

withFilter4d :: (TensorDataType a)
             => IOTensor a
             -> (CuDNN.FilterDescriptor -> CUDA.DevicePtr a -> IO b)
             -> IO b
withFilter4d tensor action = do
  datatype <- dtype tensor
  [n,c,h,w] <- shape tensor >>= return . map fromIntegral
  withDescriptor
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
    CuDNN.createConvolutionDescriptor
    (\d -> CuDNN.setConvolution2dDescriptor d cpadh cpadw cstrh cstrw cupw cuph CuDNN.convolution)
    CuDNN.destroyConvolutionDescriptor

createHandle :: IO (CuDNN.Handle)
createHandle = alloca $ \hptr -> CuDNN.createHandle hptr >> peek hptr

-- CuDNN bindings with mutable tensors.
-- convolution
convolution2dFwd :: forall m a . (PrimMonad m, TensorDataType a)
                 => CuDNN.Handle
                 -> CuDNN.ConvolutionFwdAlgo
                 -> (Int, Int)
                 -> (Int, Int)
                 -> (Int, Int)
                 -> MTensor (PrimState m) a
                 -> MTensor (PrimState m) a
                 -> m (MTensor (PrimState m) a)
convolution2dFwd handle algo padding stride upscale fmaps filters = do
  res <- unsafePrimToPrim $ (convolution2dFwdIO handle algo padding stride upscale
         (unsafeCoerce fmaps :: IOTensor a)
         (unsafeCoerce filters :: IOTensor a) :: IO (IOTensor a))
  return $ unsafeCoerce res

convolution2dFwdIO :: forall a . (TensorDataType a)
                   => CuDNN.Handle
                   -> CuDNN.ConvolutionFwdAlgo
                   -> (Int, Int)
                   -> (Int, Int)
                   -> (Int, Int)
                   -> IOTensor a
                   -> IOTensor a
                   -> IO (IOTensor a)
convolution2dFwdIO
  handle algo padding stride upscale fmaps filters = do
  -- check inputs
  fmapshape <- shape fmaps
  filtershape <- shape filters
  let shapeError = ioError $ userError $
        "Incompatible feature maps and filters shapes: "
        ++ show fmapshape
        ++ " and "
        ++ show filtershape
  fmapdims <- nbdims fmaps
  filterdims <- nbdims filters
  when (fmapdims /= 4 || filterdims /= 4) shapeError
  -- make the descriptors
  withConvDesc padding stride upscale $ \convdesc -> do
    withTensor4d fmaps $ \inputdesc inputptr -> do
      withFilter4d filters $ \filtersdesc filtersptr -> do
        -- computing output shape
        [outn,outc,outh,outw] <-
          withMany (const alloca) "bite" $ \[outnp,outcp,outhp,outwp] -> do
            handleError "Couldn't get convolution output shape." $
              CuDNN.getConvolution2dForwardOutputDim
                convdesc inputdesc filtersdesc outnp outcp outhp outwp
            forM [outnp,outcp,outhp,outwp] $ \ptr -> do
              val <- peek ptr
              return $ fromIntegral val
        output <- emptyTensor [outn,outc,outh,outw]
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
            withArray [1] $ \alpha -> withArray [1] $ \beta -> do
              -- finally run the damn thing
              handleError "Couldn't compute convolution." $
                CuDNN.convolutionForward
                handle alpha inputdesc inputptr filtersdesc filtersptr
                convdesc algo workspace workspacesize beta outputdesc
                outputptr
        return output

convolution2dBwdFilters :: forall m a . (PrimMonad m, TensorDataType a)
                         => CuDNN.Handle
                         -> (Int, Int)
                         -> (Int, Int)
                         -> (Int, Int)
                         -> MTensor (PrimState m) a
                         -> MTensor (PrimState m) a
                         -> MTensor (PrimState m) a
                         -> m (MTensor (PrimState m) a)
convolution2dBwdFilters handle padding stride upscale fmaps filters upgrad = do
  res <- unsafePrimToPrim $ convolution2dBwdFiltersIO handle padding stride upscale
         (unsafeCoerce fmaps :: IOTensor a) (unsafeCoerce filters :: IOTensor a)
         (unsafeCoerce upgrad :: IOTensor a)
  return $ unsafeCoerce res

convolution2dBwdFiltersIO :: forall a . (TensorDataType a)
                          => CuDNN.Handle
                          -> (Int, Int)
                          -> (Int, Int)
                          -> (Int, Int)
                          -> IOTensor a -- feature maps at which to evaluate gradient
                          -> IOTensor a -- filters. Only shape is used.
                          -> IOTensor a -- gradient with regards to data (upper layer)
                          -> IO (IOTensor a) -- gradients with regards to the filters
convolution2dBwdFiltersIO handle padding stride upscale fmaps filters upgrad = do
  -- make the descriptors
  withConvDesc padding stride upscale $ \convdesc -> do
    withTensor4d fmaps $ \inputdesc inputptr -> do
      withTensor4d upgrad $ \upgraddesc upgradptr -> do
        withArray [1] $ \alpha -> withArray [1] $ \beta -> do
          -- compute gradient with regards to the filters
          filtersgrad <- shape filters >>= emptyTensor
          withFilter4d filtersgrad $ \filtersgraddesc filtersgradptr ->
            handleError
            "Couldn't compute convolution gradient with respect to filters." $
            CuDNN.convolutionBackwardFilter handle alpha inputdesc inputptr
            upgraddesc upgradptr convdesc beta filtersgraddesc filtersgradptr
          return filtersgrad

convolution2dBwdInputs :: forall m a . (PrimMonad m, TensorDataType a)
                       => CuDNN.Handle
                       -> (Int, Int)
                       -> (Int, Int)
                       -> (Int, Int)
                       -> MTensor (PrimState m) a
                       -> MTensor (PrimState m) a
                       -> MTensor (PrimState m) a
                       -> m (MTensor (PrimState m) a)
convolution2dBwdInputs handle padding stride upscale fmaps filters upgrad = do
  res <- unsafePrimToPrim $ convolution2dBwdInputsIO handle padding stride upscale
         (unsafeCoerce fmaps :: IOTensor a) (unsafeCoerce filters :: IOTensor a)
         (unsafeCoerce upgrad :: IOTensor a)
  return $ unsafeCoerce res

convolution2dBwdInputsIO :: forall a . (TensorDataType a)
                         => CuDNN.Handle
                         -> (Int, Int)
                         -> (Int, Int)
                         -> (Int, Int)
                         -> IOTensor a -- feature maps input. Only shape is used.
                         -> IOTensor a -- filters at which to evaluate gradient
                         -> IOTensor a -- gradient with regards to data (upper layer)
                         -> IO (IOTensor a) -- gradients with regards to the inputs
convolution2dBwdInputsIO handle padding stride upscale fmaps filters upgrad = do
  -- make the descriptors
  withConvDesc padding stride upscale $ \convdesc -> do
    withFilter4d filters $ \filtersdesc filtersptr -> do
      withTensor4d upgrad $ \upgraddesc upgradptr -> do
        withArray [1] $ \alpha -> withArray [1] $ \beta -> do
          -- compute gradient with regards to the input feature maps
          inputsgrad <- shape fmaps >>= emptyTensor
          withTensor4d inputsgrad $ \inputsgraddesc inputsgradptr ->
            handleError
            "Couldn't compute convolution gradient with respect to the inputs." $
            CuDNN.convolutionBackwardData handle alpha filtersdesc filtersptr
            upgraddesc upgradptr convdesc beta inputsgraddesc inputsgradptr
          return inputsgrad

-- activations
activationFwd :: forall m a . (PrimMonad m, TensorDataType a)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) a
              -> m (MTensor (PrimState m) a)
activationFwd handle mode input = do
  out <- unsafePrimToPrim $
         activationFwdIO handle mode (unsafeCoerce input :: IOTensor a)
  return $ unsafeCoerce out

activationBwd :: forall m a . (PrimMonad m, TensorDataType a)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) a
              -> MTensor (PrimState m) a
              -> MTensor (PrimState m) a
              -> m (MTensor (PrimState m) a)
activationBwd handle mode input output upgrad = do
  out <- unsafePrimToPrim $
         activationBwdIO handle mode (unsafeCoerce input :: IOTensor a)
         (unsafeCoerce output :: IOTensor a) (unsafeCoerce upgrad :: IOTensor a)
  return $ unsafeCoerce out

activationFwdIO :: forall a . (TensorDataType a)
                => CuDNN.Handle
                -> CuDNN.ActivationMode
                -> IOTensor a
                -> IO (IOTensor a)
activationFwdIO handle mode input = do
  withTensor4d input $ \inputdesc inputptr -> do
    output <- shape input >>= emptyTensor
    withTensor4d output $ \outputdesc outputptr -> do
      withArray [1] $ \alphabeta -> do
        handleError "Couldn't compute activations." $
          CuDNN.activationForward handle mode alphabeta inputdesc
          inputptr alphabeta outputdesc outputptr
        return output

activationBwdIO :: forall a . (TensorDataType a)
                => CuDNN.Handle
                -> CuDNN.ActivationMode
                -> IOTensor a
                -> IOTensor a -- output of the forward pass on the input
                -> IOTensor a
                -> IO (IOTensor a)
activationBwdIO handle mode input output upgrad = do
  withTensor4d input $ \inputdesc inputptr -> do
    withTensor4d upgrad $ \upgraddesc upgradptr -> do
      grad <- shape input >>= emptyTensor
      withTensor4d output $ \outputdesc outputptr -> do
        withTensor4d grad $ \graddesc gradptr -> do
          withArray [1] $ \alphabeta -> do
            handleError "Couldn't compute activation backward pass." $
              CuDNN.activationBackward handle mode alphabeta inputdesc inputptr
              upgraddesc upgradptr outputdesc outputptr alphabeta graddesc
              gradptr
            return grad

-- 2d pooling
pooling2dFwd :: forall m a . (PrimMonad m, TensorDataType a)
           => CuDNN.Handle
           -> CuDNN.PoolingMode
           -> (Int, Int)
           -> (Int, Int)
           -> (Int, Int)
           -> MTensor (PrimState m) a
           -> m (MTensor (PrimState m) a)
pooling2dFwd handle mode size padding stride input = do
  out <- unsafePrimToPrim $ pooling2dFwdIO handle mode size padding stride
         (unsafeCoerce input :: IOTensor a)
  return $ unsafeCoerce out

pooling2dBwd :: forall m a . (PrimMonad m, TensorDataType a)
           => CuDNN.Handle
           -> CuDNN.PoolingMode
           -> (Int, Int)
           -> (Int, Int)
           -> (Int, Int)
           -> MTensor (PrimState m) a
           -> MTensor (PrimState m) a
           -> MTensor (PrimState m) a
           -> m (MTensor (PrimState m) a)
pooling2dBwd handle mode size padding stride inp out upgrad = do
  grad <- unsafePrimToPrim $ pooling2dBwdIO handle mode size padding stride
          (unsafeCoerce inp :: IOTensor a) (unsafeCoerce out :: IOTensor a)
          (unsafeCoerce upgrad :: IOTensor a)
  return $ unsafeCoerce grad

pooling2dFwdIO :: forall a . (TensorDataType a)
             => CuDNN.Handle
             -> CuDNN.PoolingMode
             -> (Int, Int)
             -> (Int, Int)
             -> (Int, Int)
             -> IOTensor a
             -> IO (IOTensor a)
pooling2dFwdIO handle mode (wh,ww) (padh,padw) (strh,strw) input = do
  -- pooling descriptor
  let [cwh,cww,cpadh,cpadw,cstrh,cstrw] =
        map fromIntegral [wh,ww,padh,padw,strh,strw]
  withDescriptor
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d input $ \inputdesc inputptr -> do
        -- output shape
        outshape <- withMany (const alloca) "bite" $ \[outnp,outcp,outhp,outwp] -> do
          handleError "Couldn't compute pooling output shape." $
            CuDNN.getPooling2dForwardOutputDim pooldesc inputdesc outnp outcp outhp
            outwp
          forM [outnp,outcp,outhp,outwp] peek
        output <- emptyTensor $ map fromIntegral outshape
        -- actual pooling
        withTensor4d output $ \outputdesc outputptr -> do
          withArray [1] $ \alphabeta -> do
            handleError "Couldn't compute pooling." $
              CuDNN.poolingForward handle pooldesc alphabeta inputdesc
              inputptr alphabeta outputdesc outputptr
            return output

pooling2dBwdIO :: forall a . (TensorDataType a)
             => CuDNN.Handle
             -> CuDNN.PoolingMode
             -> (Int, Int)
             -> (Int, Int)
             -> (Int, Int)
             -> IOTensor a
             -> IOTensor a
             -> IOTensor a
             -> IO (IOTensor a)
pooling2dBwdIO handle mode (wh,ww) (padh,padw) (strh,strw) inp out upgrad = do
  -- pooling descriptor
  let [cwh,cww,cpadh,cpadw,cstrh,cstrw] =
        map fromIntegral [wh,ww,padh,padw,strh,strw]
  withDescriptor
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d inp $ \inpdesc inpptr -> do
        withTensor4d out $ \outdesc outptr -> do
          withTensor4d upgrad $ \upgraddesc upgradptr -> do
            grad <- shape inp >>= emptyTensor
            withTensor4d grad $ \graddesc gradptr -> do
              withArray [1] $ \alphabeta -> do
                handleError "Couldn't compute backward pooling." $
                  CuDNN.poolingBackward handle pooldesc alphabeta inpdesc inpptr
                  upgraddesc upgradptr outdesc outptr alphabeta graddesc gradptr
                return grad

-- Softmax
softmaxFwd :: forall m a . (PrimMonad m, TensorDataType a)
           => CuDNN.Handle
           -> CuDNN.SoftmaxAlgorithm
           -> CuDNN.SoftmaxMode
           -> MTensor (PrimState m) a
           -> m (MTensor (PrimState m) a)
softmaxFwd handle algo mode input = do
  res <- unsafePrimToPrim $ softmaxFwdIO handle algo mode
         (unsafeCoerce input :: IOTensor a)
  return $ unsafeCoerce res

softmaxBwd :: forall m a . (PrimMonad m, TensorDataType a)
           => CuDNN.Handle
           -> CuDNN.SoftmaxAlgorithm
           -> CuDNN.SoftmaxMode
           -> MTensor (PrimState m) a
           -> MTensor (PrimState m) a
           -> m (MTensor (PrimState m) a)
softmaxBwd handle algo mode src srcdiff = do
  res <- unsafePrimToPrim $ softmaxBwdIO handle algo mode
         (unsafeCoerce src :: IOTensor a) (unsafeCoerce srcdiff :: IOTensor a)
  return $ unsafeCoerce res

softmaxFwdIO :: (TensorDataType a)
             => CuDNN.Handle
             -> CuDNN.SoftmaxAlgorithm
             -> CuDNN.SoftmaxMode
             -> IOTensor a
             -> IO (IOTensor a)
softmaxFwdIO handle algo mode input = do
  withTensor4d input $ \inpdesc inpptr -> do
    output <- shape input >>= emptyTensor
    withTensor4d output $ \outdesc outptr -> do
      withArray [1] $ \alphabeta -> do
        handleError "Couldn't compute softmax." $
          CuDNN.softmaxForward handle algo mode alphabeta
          inpdesc inpptr alphabeta outdesc outptr
        return output

softmaxBwdIO :: (TensorDataType a)
             => CuDNN.Handle
             -> CuDNN.SoftmaxAlgorithm
             -> CuDNN.SoftmaxMode
             -> IOTensor a
             -> IOTensor a
             -> IO (IOTensor a)
softmaxBwdIO handle algo mode src srcdiff = do
  withTensor4d src $ \srcdesc srcdata -> do
    withTensor4d srcdiff $ \srcdiffdesc srcdiffdata -> do
      destdiff <- shape src >>= emptyTensor
      withTensor4d destdiff $ \destdiffdesc destdiffdata -> do
        withArray [1] $ \alphabeta -> do
          handleError "Couldn't compute softmax backward pass." $
            CuDNN.softmaxBackward handle algo mode alphabeta
            srcdesc srcdata srcdiffdesc srcdiffdata alphabeta
            destdiffdesc destdiffdata
          return destdiff

-- Utility functions leveraging CuBlas.
-- GEMM wrapper, used to implement roughly everything else.
-- Computes alpha * dot(A,B) + beta * C, puts the results in C.
gemmFwdIO :: (TensorDataType a)
          => Cublas.Handle
          -> a
          -> IOTensor a
          -> IOTensor a
          -> a
          -> IOTensor a
          -> IO ()
gemmFwdIO handle alpha a b beta c = do
  -- shape information
  [m, k] <- shape a
  [k', n] <- shape b
  [m', n'] <- shape c
  when (m /= m' || n /= n' || k /= k') $ ioError $ userError $
    "Incompatible shapes for GEMM: " ++ show [m,k] ++ " "
    ++ show [k',n] ++ " " ++ show [m',n']
  withDevicePtr a $ \aptr -> do
    withDevicePtr b $ \bptr -> do
      withDevicePtr c $ \cptr -> do
        Cublas.gemm handle Cublas.N Cublas.N m n k alpha
          aptr m bptr k beta cptr m

-- GEMM backward pass with regards to A
gemmBwdAIO :: (TensorDataType a)
           => Cublas.Handle
           -> a
           -> IOTensor a
           -> IOTensor a
           -> IO (IOTensor a)
gemmBwdAIO handle alpha b upgrad = do
  -- shape information
  [k, n] <- shape b
  [m, n'] <- shape upgrad
  when (n /= n') $ ioError $ userError $
    "Incompatible shapes for backward GEMM for A: "
    ++ show [k,n] ++ " " ++ show [m,n']
  out <- emptyTensor [m, k]
  withDevicePtr b $ \bptr -> do
    withDevicePtr upgrad $ \upgradptr -> do
      withDevicePtr out $ \outptr -> do
        Cublas.gemm handle Cublas.N Cublas.T m k n alpha
          upgradptr m bptr n 0 outptr m
        return out

-- GEMM backward pass with regards to B
gemmBwdBIO :: (TensorDataType a)
           => Cublas.Handle
           -> a
           -> IOTensor a
           -> IOTensor a
           -> IO (IOTensor a)
gemmBwdBIO handle alpha a upgrad = do
  [m, k] <- shape a
  [m', n] <- shape upgrad
  when (m /= m') $ ioError $ userError $
    "Incompatible shapes for backward GEMM for B: "
    ++ show [m,k] ++ " " ++ show [m',n]
  out <- emptyTensor [k, n]
  withDevicePtr a $ \aptr -> do
    withDevicePtr upgrad $ \upgradptr -> do
      withDevicePtr out $ \outptr -> do
        Cublas.gemm handle Cublas.T Cublas.N k n m alpha
          aptr k upgradptr m 0 outptr k
        return out

-- GEMM backward pass with regards to C
gemmBwdCIO :: (TensorDataType a)
           => Cublas.Handle
           -> a
           -> IOTensor a
           -> IO (IOTensor a)
gemmBwdCIO handle beta upgrad = do
  out <- copy upgrad
  shp <- shape upgrad
  withDevicePtr upgrad $ \upgradptr -> do
    Cublas.scal handle (product shp) beta upgradptr 1
    return out
