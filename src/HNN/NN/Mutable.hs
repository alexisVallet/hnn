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
  , softmaxFwd
  , softmaxBwd
  , gemmFwd
  , gemmBwdA
  , gemmBwdB
  , gemmBwdC
  , createGenerator
  , dropoutMaskIO
  , transformTensor
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

withTensor4d :: (TensorDataType a)
             => IOTensor a
             -> (CuDNN.TensorDescriptor -> CUDA.DevicePtr a -> IO b)
             -> IO b
withTensor4d tensor action = do
  datatype <- dtype tensor
  [n,c,h,w] <- shape tensor >>= return . map fromIntegral
  withDescriptor
    "tensor"
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
  let runConv = withConvDesc padding stride upscale $ \convdesc -> do
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
                withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                  -- finally run the damn thing
                  handleError "Couldn't compute convolution." $
                    CuDNN.convolutionForward
                    handle alpha inputdesc inputptr filtersdesc filtersptr
                    convdesc algo workspace workspacesize beta outputdesc
                    outputptr
            return output
  runConv `catch` \e -> do
    error $ "Exception thrown in convolution forward pass: " ++ show (e :: SomeException) ++ "\n filter shape: " ++ show filtershape ++ ", image shape: " ++ show fmapshape

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
  let runConv = withConvDesc padding stride upscale $ \convdesc -> do
        withTensor4d fmaps $ \inputdesc inputptr -> do
          withTensor4d upgrad $ \upgraddesc upgradptr -> do
            withArray [1] $ \alpha -> withArray [0] $ \beta -> do
              -- compute gradient with regards to the filters
              filtersgrad <- shape filters >>= emptyTensor
              withFilter4d filtersgrad $ \filtersgraddesc filtersgradptr ->
                handleError
                "Couldn't compute convolution gradient with respect to filters." $
                CuDNN.convolutionBackwardFilter handle alpha inputdesc inputptr
                upgraddesc upgradptr convdesc beta filtersgraddesc filtersgradptr
              return filtersgrad
  runConv `catch` \e -> error $ "Exception thrown in convolution backward pass with regards to filters: " ++ show (e :: IOException)

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
  let runConv = withConvDesc padding stride upscale $ \convdesc -> do
        withFilter4d filters $ \filtersdesc filtersptr -> do
          withTensor4d upgrad $ \upgraddesc upgradptr -> do
            withArray [1] $ \alpha -> withArray [0] $ \beta -> do
              -- compute gradient with regards to the input feature maps
              inputsgrad <- shape fmaps >>= emptyTensor
              withTensor4d inputsgrad $ \inputsgraddesc inputsgradptr ->
                handleError
                "Couldn't compute convolution gradient with respect to the inputs." $
                CuDNN.convolutionBackwardData handle alpha filtersdesc filtersptr
                upgraddesc upgradptr convdesc beta inputsgraddesc inputsgradptr
              return inputsgrad
  runConv `catch` \e -> error $ "Exception thrown in convolution backward pass with regards to data: " ++ show (e :: IOException)
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
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute activations." $
          CuDNN.activationForward handle mode alpha inputdesc
          inputptr beta outputdesc outputptr
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
    "pooling"
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d input $ \inputdesc inputptr -> do
        -- output shape
        inshape <- shape input
        let
          outshape = pooling2dOutputShape (wh,ww) (padh,padw) (strh,strw) inshape
        output <- emptyTensor outshape
        -- actual pooling
        withTensor4d output $ \outputdesc outputptr -> do
          withArray [1] $ \alpha -> withArray [0] $ \beta -> do
            handleError "Couldn't compute pooling." $
              CuDNN.poolingForward handle pooldesc alpha inputdesc
              inputptr beta outputdesc outputptr
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
    "pooling"
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d inp $ \inpdesc inpptr -> do
        withTensor4d out $ \outdesc outptr -> do
          withTensor4d upgrad $ \upgraddesc upgradptr -> do
            grad <- shape inp >>= emptyTensor
            withTensor4d grad $ \graddesc gradptr -> do
              withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                handleError "Couldn't compute backward pooling." $
                  CuDNN.poolingBackward handle pooldesc alpha outdesc outptr
                  upgraddesc upgradptr inpdesc inpptr beta graddesc gradptr
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
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute softmax." $
          CuDNN.softmaxForward handle algo mode alpha
          inpdesc inpptr beta outdesc outptr
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
gemmFwd :: forall m a . (PrimMonad m, TensorDataType a)
        => Cublas.Handle
        -> Cublas.Operation
        -> Cublas.Operation
        -> a
        -> MTensor (PrimState m) a
        -> MTensor (PrimState m) a
        -> a
        -> MTensor (PrimState m) a
        -> m ()
gemmFwd handle transa transb alpha a b beta c = do
  unsafePrimToPrim $ gemmFwdIO handle transa transb alpha
    (unsafeCoerce a :: IOTensor a) (unsafeCoerce b :: IOTensor b) beta
    (unsafeCoerce c :: IOTensor a)

-- Composes 2 cublas operations into one.
compop :: Cublas.Operation -> Cublas.Operation -> Cublas.Operation
compop Cublas.N op = op
compop op Cublas.N = op
compop Cublas.T Cublas.T = Cublas.N
compop op1 op2 = error $ "Unsupported operations: " ++ show (op1, op2)

gemmBwdA :: (PrimMonad m, TensorDataType a)
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) a
         -> MTensor (PrimState m) a
         -> MTensor (PrimState m) a
         -> m ()
gemmBwdA handle transa transb alpha b upgrad out = do
  -- Need to distinguish between the case where
  -- we have an operation on A or not.
  case transa of
    Cublas.N -> gemmFwd handle Cublas.N (compop transb Cublas.T)
                alpha upgrad b 0 out
    Cublas.T -> gemmFwd handle transb Cublas.T alpha b upgrad 0 out

gemmBwdB :: (PrimMonad m, TensorDataType a)
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) a
         -> MTensor (PrimState m) a
         -> MTensor (PrimState m) a
         -> m ()
gemmBwdB handle transa transb alpha a upgrad out = do
  case transb of
    Cublas.N -> gemmFwd handle (compop transa Cublas.T) Cublas.N alpha
                a upgrad 0 out
    Cublas.T -> gemmFwd handle Cublas.T transa alpha upgrad a 0 out

gemmBwdC :: forall m a . (PrimMonad m, TensorDataType a)
         => Cublas.Handle
         -> a
         -> MTensor (PrimState m) a
         -> MTensor (PrimState m) a
         -> m ()
gemmBwdC handle beta upgrad out = do
  unsafePrimToPrim $ gemmBwdCIO handle beta
    (unsafeCoerce upgrad :: IOTensor a) (unsafeCoerce out :: IOTensor a)
  
gemmBwdCIO :: (TensorDataType a)
           => Cublas.Handle
           -> a
           -> IOTensor a
           -> IOTensor a
           -> IO ()
gemmBwdCIO handle beta upgrad out = do
  size <- fmap product $ shape upgrad
  withDevicePtr upgrad $ \upgradptr -> do
    withDevicePtr out $ \outptr -> do
      Cublas.copy handle size upgradptr 1 outptr 1
      Cublas.scal handle size beta outptr 1
  
gemmFwdIO :: (TensorDataType a)
          => Cublas.Handle
          -> Cublas.Operation
          -> Cublas.Operation
          -> a
          -> IOTensor a
          -> IOTensor a
          -> a
          -> IOTensor a
          -> IO ()
gemmFwdIO handle transa transb alpha a b beta c = do
  -- Figuring out the parameters to pass GEMM so it does
  -- the right thing in row-major layout, depending on the
  -- user requested transforms.
  [ra, ca] <- shape a
  [rb, cb] <- shape b
  [rc, cc] <- shape c
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
dropoutMaskIO :: TensorDataType a
              => CuRAND.Generator
              -> a
              -> [Int]
              -> IO (IOTensor a)
dropoutMaskIO gen drop_proba shp = do
  -- Simple algo for dropout of activations:
  -- * generate an array of random values between 0 and 1
  -- * threshold that array with the dropout probability
  -- * elementwise multiply the input array with it
  rand_array <- emptyTensor shp
  let size = fromIntegral $ product shp
  withDevicePtr rand_array $ \randarrayptr -> do
    -- generate random array
    generateUniform gen randarrayptr size
    -- threshold it
    thresh randarrayptr size drop_proba randarrayptr
    return rand_array

-- element-wise multiply (in place)
mul :: forall a m . (TensorDataType a, PrimMonad m)
    => MTensor (PrimState m) a
    -> MTensor (PrimState m) a
    -> m ()
mul t1 t2 =
  unsafePrimToPrim $ mulIO
  (unsafeCoerce t1 :: IOTensor a) (unsafeCoerce t2 :: IOTensor a)

mulIO :: TensorDataType a
      => IOTensor a
      -> IOTensor a
      -> IO ()
mulIO t1 t2 = do
  size <- fmap (fromIntegral . product) $ shape t1
  withDevicePtr t1 $ \t1ptr ->
    withDevicePtr t2 $ \t2ptr -> do
      rawMul t1ptr t2ptr size

transformTensorIO :: TensorDataType a
                  => CuDNN.Handle
                  -> CuDNN.TensorFormat
                  -> CuDNN.TensorFormat
                  -> IOTensor a
                  -> IO (IOTensor a)
transformTensorIO handle srcf dstf src = do
  datatype <- dtype src
  [n,c,h,w] <- shape src
               >>= \[a,b,c,d] -> return $
                                 if srcf == CuDNN.nchw then [a,b,c,d]
                                 else if srcf == CuDNN.nhwc then [a,d,b,c]
                                      else error $ "Unsupported input format: " ++ show srcf
  let [cn,cc,ch,cw] = fmap fromIntegral [n,c,h,w]
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
          dst <- emptyTensor $ if dstf == CuDNN.nchw then [n,c,h,w]
                               else if dstf == CuDNN.nhwc then [n,h,w,c]
                                    else error $ "Unsupported output format: " ++ show dstf
          withDevicePtr src $ \srcptr -> do
            withDevicePtr dst $ \dstptr -> do
              withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                handleError "Couldn't transform the tensor." $
                  CuDNN.transformTensor handle alpha srcdesc srcptr beta
                  dstdesc dstptr
                return dst

transformTensor :: forall a m . (TensorDataType a, PrimMonad m)
                => CuDNN.Handle
                -> CuDNN.TensorFormat
                -> CuDNN.TensorFormat
                -> MTensor (PrimState m) a
                -> m (MTensor (PrimState m) a)
transformTensor handle srcf dstf src = do
  ioout <- unsafePrimToPrim
           $ transformTensorIO  handle srcf dstf (unsafeCoerce src :: IOTensor a)
  return $ unsafeCoerce ioout

addTensorIO :: (TensorDataType a)
            => CuDNN.Handle
            -> CuDNN.AddMode
            -> IOTensor a
            -> IOTensor a
            -> IO (IOTensor a)
addTensorIO handle addMode bias src = do
  out <- copy src
  withTensor4d bias $ \biasdesc biasptr -> do
    withTensor4d out $ \outdesc outptr -> do
      withArray [1] $ \alpha -> withArray [1] $ \beta -> do
        handleError "Couldn't add the tensors." $
          CuDNN.addTensor handle addMode alpha biasdesc biasptr
          beta outdesc outptr
  return out

addTensor :: forall a m . (TensorDataType a, PrimMonad m)
          => CuDNN.Handle
          -> CuDNN.AddMode
          -> MTensor (PrimState m) a
          -> MTensor (PrimState m) a
          -> m (MTensor (PrimState m) a)
addTensor handle addMode bias src = do
  out <- unsafePrimToPrim $ addTensorIO handle addMode
         (unsafeCoerce bias :: IOTensor a) (unsafeCoerce src :: IOTensor a)
  return $ unsafeCoerce out

convolutionBackwardBiasIO :: (TensorDataType a)
                          => CuDNN.Handle
                          -> IOTensor a
                          -> IO (IOTensor a)
convolutionBackwardBiasIO handle upgrad = do
  [n,c,h,w] <- shape upgrad
  biasgrad <- emptyTensor [1,c,1,1]
  withTensor4d upgrad $ \upgraddesc upgradptr -> do
    withTensor4d biasgrad $ \biasgraddesc biasgradptr -> do
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute convolution backward pass with regards to bias" $
          CuDNN.convolutionBackwardBias handle alpha upgraddesc upgradptr
          beta biasgraddesc biasgradptr
  return biasgrad

convolutionBackwardBias :: forall a m . (TensorDataType a, PrimMonad m)
                        => CuDNN.Handle
                        -> MTensor (PrimState m) a
                        -> m (MTensor (PrimState m) a)
convolutionBackwardBias handle upgrad = do
  biasgrad <- unsafePrimToPrim $ convolutionBackwardBiasIO handle
              (unsafeCoerce upgrad :: IOTensor a)
  return $ unsafeCoerce biasgrad

