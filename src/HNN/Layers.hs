{-# LANGUAGE TemplateHaskell #-}
module HNN.Layers (
    LayerMonad
  , convolution2d
  , activation
  , pooling2d
  , dropout
  , runLayer
  , CuDNN.ConvolutionFwdAlgo
  , CuDNN.ActivationMode
  , CuDNN.PoolingMode
  ) where

import Foreign.C
import Foreign.Marshal
import Control.Monad.Reader
import Control.Monad.RWS
import Control.Monad.ST
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.CuRAND as CuRAND
import Control.Lens
import Control.Monad.Primitive
import System.IO.Unsafe

import HNN.Tensor
import HNN.NN
import HNN.NN.Mutable

data LayerReader = LayerReader {
  _cublasHandle :: Cublas.Handle,
  _cudnnHandle :: CuDNN.Handle,
  _generator :: CuRAND.Generator
  }
makeLenses ''LayerReader

type LayerMonad  = ReaderT LayerReader IO

type Layer a w = NN LayerMonad a w (Tensor a) (Tensor a)

convolution2d :: (TensorDataType a)
              => CuDNN.ConvolutionFwdAlgo
              -> (Int, Int)
              -> (Int, Int)
              -> (Int, Int)
              -> Layer a (Tensor a)
convolution2d algo padding stride upscale =
  fromFwdBwd convfwd convbwd
  where convfwd filters fmaps = do
          handle <- view cudnnHandle
          return $ runST $ do
            filters' <- unsafeThaw filters
            fmaps' <- unsafeThaw fmaps
            convres <- convolution2dFwd handle algo padding
                       stride upscale filters' fmaps'
            unsafeFreeze convres
        convbwd filters fmaps _ = do
          handle <- view cudnnHandle
          let bwdfilters upgrad = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                filtersgrad <- convolution2dBwdFilters handle padding
                               stride upscale fmaps' filters' upgrad'
                unsafeFreeze filtersgrad
              bwdinputs upgrad = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                inputsgrad <- convolution2dBwdInputs handle padding stride
                              upscale fmaps' filters' upgrad'
                unsafeFreeze inputsgrad
          return $ \upgrad -> (bwdfilters upgrad, bwdinputs upgrad)

activation :: (TensorDataType a)
           => CuDNN.ActivationMode
           -> Layer a (Trivial a)
activation mode =
  fromFwdBwd actfwd actbwd
  where actfwd Zero fmaps = do
          handle <- view cudnnHandle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            activations <- activationFwd handle mode fmaps'
            unsafeFreeze activations
        actbwd Zero inp out = do
          handle <- view cudnnHandle
          return $ \upgrad -> runST $ do
            inp' <- unsafeThaw inp
            out' <- unsafeThaw out
            upgrad' <- unsafeThaw upgrad
            grad <- activationBwd handle mode inp' out' upgrad'
            grad' <- unsafeFreeze grad
            return (Zero, grad')

-- pooling
pooling2d :: (TensorDataType a)
          => CuDNN.PoolingMode
          -> (Int, Int)
          -> (Int, Int)
          -> (Int, Int)
          -> Layer a (Trivial a)
pooling2d mode size padding stride =
  fromFwdBwd poolfwd poolbwd
  where poolfwd Zero fmaps = do
          handle <- view cudnnHandle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            poolres <- pooling2dFwd handle mode size padding stride fmaps'
            unsafeFreeze poolres
        poolbwd Zero inp out = do
          handle <- view cudnnHandle
          return $ \upgrad -> runST $ do
            [inp', out', upgrad'] <- forM [inp, out, upgrad] unsafeThaw
            grad <- pooling2dBwd handle mode size padding stride
                    inp' out' upgrad'
            grad' <- unsafeFreeze grad
            return (Zero, grad')

-- dropout
dropout :: (TensorDataType a)
        => a
        -> Layer a (Trivial a)
dropout drop_proba = NN fwdbwd
  -- Simple dropout algo: generate a random tensor of 0 and 1s,
  -- elementwise multiply with the same random tensor on both forward
  -- and backward pass.
  where fwdbwd Zero inp = do
          gen <- view generator
          mask <- liftIO $ dropoutMaskIO gen drop_proba (shape inp)
          pure_mask <- unsafeFreeze mask
          return (inp * pure_mask, \upgrad -> (Zero, upgrad * pure_mask))

-- Actually running the thing.
runLayer :: CULLong -> LayerMonad a -> IO a
runLayer rngSeed action = do
  cudnn <- createHandle
  cublas <- Cublas.create
  curand <- createGenerator CuRAND.rng_pseudo_default
  CuRAND.setPseudoRandomGeneratorSeed curand rngSeed
  runReaderT action $ LayerReader cublas cudnn curand
