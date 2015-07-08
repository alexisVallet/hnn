{-# LANGUAGE ScopedTypeVariables #-}
module HNN.NN where

import Foreign
import Foreign.C
import Foreign.Marshal
import Control.Monad
import Control.Monad.Reader
import Data.Proxy
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN

import HNN.Tensor

type HNNMonad = ReaderT CuDNN.Handle IO

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

createHandle :: IO (CuDNN.Handle)
createHandle = alloca $ \hptr -> CuDNN.createHandle hptr >> peek hptr

convolution2dFwd :: forall a . (Storable a, TensorDataType a)
                 => CuDNN.Handle
                 -> CuDNN.ConvolutionFwdAlgo
                 -> (Int, Int)
                 -> (Int, Int)
                 -> (Int, Int)
                 -> IOTensor a
                 -> IOTensor a
                 -> IO (IOTensor a)
convolution2dFwd
  handle algo (padh,padw) (strh,strw) (upw,uph) fmaps filters = do
  -- check inputs  
  let shapeError = ioError $ userError $
        "Incompatible feature maps and filters shapes: "
        ++ show (shape fmaps)
        ++ " and "
        ++ show (shape filters)
  when (nbdims fmaps /= 4 || nbdims filters /= 4) shapeError
  let [n,c,h,w] = map fromIntegral $ shape fmaps
      [fn,fc,fh,fw] = map fromIntegral $ shape filters
      dtype = datatype (Proxy :: Proxy a)
      [cpadh,cpadw,cstrh,cstrw,cupw,cuph] =
        map fromIntegral [padh,padw,strh,strw,upw,uph]
  -- make the descriptors
  putStrLn "Making source tensor descriptor..."
  withDescriptor
    CuDNN.createTensorDescriptor
    (\desc -> CuDNN.setTensor4dDescriptor desc CuDNN.nchw dtype n c h w)
    CuDNN.destroyTensorDescriptor $ \inputdesc -> do
    putStrLn "Making filters descriptor..."
    withDescriptor
      CuDNN.createFilterDescriptor
      (\desc -> CuDNN.setFilter4dDescriptor desc dtype fn fc fh fw)
      CuDNN.destroyFilterDescriptor $ \filtersdesc -> do
      putStrLn "Making convolution descriptor..."
      withDescriptor
        CuDNN.createConvolutionDescriptor
        (\desc -> CuDNN.setConvolution2dDescriptor
                  desc cpadh cpadw cstrh cstrw cuph cupw CuDNN.convolution)
        CuDNN.destroyConvolutionDescriptor $ \convdesc -> do
          putStrLn "Computing output shape..."
          -- computing output shape
          [outn,outc,outh,outw] <-
            withMany (const alloca) "bite" $ \[outnp,outcp,outhp,outwp] -> do
              handleError "Couldn't get convolution output shape." $
                CuDNN.getConvolution2dForwardOutputDim
                  convdesc inputdesc filtersdesc outnp outcp outhp outwp
              forM [outnp,outcp,outhp,outwp] peek
          putStrLn "Making output tensor descriptor..."
          withDescriptor
            CuDNN.createTensorDescriptor
            (\desc -> CuDNN.setTensor4dDescriptor
                      desc CuDNN.nchw dtype outn outc outh outw)
            CuDNN.destroyTensorDescriptor $ \outputdesc -> do
              -- allocate workspace
              putStrLn "Computing workspace size..."
              workspacesize <- alloca $ \wkspcsizeptr -> do
                handleError "Couldn't compute workspace size." $
                  CuDNN.getConvolutionForwardWorkspaceSize
                    handle inputdesc filtersdesc convdesc outputdesc algo wkspcsizeptr
                peek wkspcsizeptr
              putStrLn "Allocating workspace..."
              CUDA.allocaArray (fromIntegral workspacesize) $ \workspace -> do
                -- allocate alpha and beta
                putStrLn "Allocating alpha and beta..."
                withArray [1] $ \alpha -> withArray [1] $ \beta -> do
                  output <- emptyTensor (map fromIntegral [outn,outc,outh,outw])
                  -- get raw device pointers to tensor data
                  withMany withDevicePtr [fmaps,filters,output] $
                    \[inpptr,filptr,outptr] -> do
                      -- finally run the damn thing
                      putStrLn "Computing convolution..."
                      handleError "Couldn't compute convolution." $
                        CuDNN.convolutionForward
                          handle alpha inputdesc inpptr filtersdesc filptr convdesc
                          algo workspace workspacesize beta outputdesc outptr
                  return output
