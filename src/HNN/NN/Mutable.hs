{-# LANGUAGE GADTs, ScopedTypeVariables #-}
module HNN.NN.Mutable (
  createHandle
  , convolution2dFwd
  , convolution2dBwdFilters
  , convolution2dBwdInputs
  ) where

import Foreign
import Foreign.C
import Foreign.Marshal
import Control.Monad
import Data.Proxy
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
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

withTensor4d :: (TensorDataType a, Storable a)
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
      

withFilter4d :: (TensorDataType a, Storable a)
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

-- bindings for cudnn functions taking care of allocation and marshaling.
convolution2dFwd :: forall m a . (PrimMonad m, Storable a, TensorDataType a)
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

convolution2dFwdIO :: forall a . (Storable a, TensorDataType a)
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

convolution2dBwdFilters :: forall m a . (PrimMonad m, Storable a, TensorDataType a)
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

convolution2dBwdFiltersIO :: forall a . (Storable a, TensorDataType a)
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

convolution2dBwdInputs :: forall m a . (PrimMonad m, Storable a, TensorDataType a)
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

convolution2dBwdInputsIO :: forall a . (Storable a, TensorDataType a)
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
