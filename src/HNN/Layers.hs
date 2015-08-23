{-# LANGUAGE TemplateHaskell #-}
module HNN.Layers (
  convolution2d
  , activation
  , pooling2d
  , CuDNN.ConvolutionFwdAlgo
  , CuDNN.ActivationMode
  , CuDNN.PoolingMode
  ) where

import Control.Monad.Reader
import Control.Monad.ST
import qualified Foreign.CUDA.CuDNN as CuDNN
import Control.Lens
import Control.Monad.Primitive

import HNN.Tensor
import HNN.NN
import HNN.NN.Mutable

data LayerReader = LayerReader {
  _handle :: CuDNN.Handle
  }
makeLenses ''LayerReader

type LayerMonad = Reader LayerReader

convolution2d :: (TensorDataType s)
              => CuDNN.ConvolutionFwdAlgo
              -> (Int, Int)
              -> (Int, Int)
              -> (Int, Int)
              -> Tensor s -- initial filters
              -> NN LayerMonad s (Tensor s) (Tensor s)
convolution2d algo padding stride upscale initfilters =
  fromFwdBwd convfwd convbwd initfilters
  where convfwd filters fmaps = do
          handle <- view handle
          return $ runST $ do
            filters' <- unsafeThaw filters
            fmaps' <- unsafeThaw fmaps
            convres <- convolution2dFwd handle algo padding
                       stride upscale filters' fmaps'
            unsafeFreeze convres
        convbwd filters fmaps _ upgrad = do
          handle <- view handle
          let bwdfilters = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                filtersgrad <- convolution2dBwdFilters handle padding
                               stride upscale fmaps' filters' upgrad'
                unsafeFreeze filtersgrad
              bwdinputs = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                inputsgrad <- convolution2dBwdInputs handle padding stride
                              upscale fmaps' filters' upgrad'
                unsafeFreeze inputsgrad
          return (bwdfilters, bwdinputs)

activation :: (TensorDataType s)
           =>  CuDNN.ActivationMode
           -> NN LayerMonad s (Tensor s) (Tensor s)
activation mode =
  fromFwdBwd actfwd actbwd ()
  where actfwd () fmaps = do
          handle <- view handle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            activations <- activationFwd handle mode fmaps'
            unsafeFreeze activations
        actbwd () inp out upgrad = do
          handle <- view handle
          return $ runST $ do
            inp' <- unsafeThaw inp
            out' <- unsafeThaw out
            upgrad' <- unsafeThaw upgrad
            grad <- activationBwd handle mode inp' out' upgrad'
            grad' <- unsafeFreeze grad
            return ((), grad')

-- Currently, some cudnn pooling utilities are broken, so until
-- I write a workaround no pooling.
pooling2d :: (TensorDataType s)
          => CuDNN.PoolingMode
          -> (Int, Int)
          -> (Int, Int)
          -> (Int, Int)
          -> NN LayerMonad s (Tensor s) (Tensor s)
pooling2d mode size padding stride =
  fromFwdBwd poolfwd poolbwd ()
  where poolfwd () fmaps = do
          handle <- view handle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            poolres <- pooling2dFwd handle mode size padding stride fmaps'
            unsafeFreeze poolres
        poolbwd () inp out upgrad = do
          handle <- view handle
          return $ runST $ do
            [inp', out', upgrad'] <- forM [inp, out, upgrad] unsafeThaw
            grad <- pooling2dBwd handle mode size padding stride
                    inp' out' upgrad'
            grad' <- unsafeFreeze grad
            return ((), grad')
