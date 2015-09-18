{-# LANGUAGE TemplateHaskell, ScopedTypeVariables #-}
module HNN.Layers (
    LayerMonad
  , Layer
  , convolution2d
  , CuDNN.convolution_fwd_algo_implicit_gemm
  , CuDNN.convolution_fwd_algo_implicit_precomp_gemm
  , CuDNN.convolution_fwd_algo_gemm
  , CuDNN.convolution_fwd_algo_direct
  , activation
  , CuDNN.activation_sigmoid
  , CuDNN.activation_relu
  , CuDNN.activation_tanh
  , pooling2d
  , CuDNN.pooling_max
  , CuDNN.pooling_average_count_include_padding
  , CuDNN.pooling_average_count_exclude_padding
  , dropout
  , linear
  , sumCols
  , sumRows
  , runLayer
  , CuDNN.ConvolutionFwdAlgo
  , CuDNN.ActivationMode
  , CuDNN.PoolingMode
  ) where
import Prelude hiding (log, (.), id)
import Foreign.C
import Foreign.Marshal
import Control.Monad.Reader
import Control.Monad.RWS
import Control.Monad.ST
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.CuRAND as CuRAND
import Control.Lens
import Control.Category
import Control.Monad.Primitive
import System.IO.Unsafe

import HNN.Tensor
import HNN.NN
import HNN.NN.Mutable
import qualified HNN.Tensor.Mutable as MT

data LayerReader = LayerReader {
  _cublasHandle :: Cublas.Handle,
  _cudnnHandle :: CuDNN.Handle,
  _generator :: CuRAND.Generator
  }
makeLenses ''LayerReader

type LayerMonad  = ReaderT LayerReader IO

type Layer a w = NN LayerMonad a w (Tensor a) (Tensor a)

-- Actually running the thing.
runLayer :: CULLong -> LayerMonad a -> IO a
runLayer rngSeed action = do
  cudnn <- createHandle
  cublas <- Cublas.create
  curand <- createGenerator CuRAND.rng_pseudo_default
  CuRAND.setPseudoRandomGeneratorSeed curand rngSeed
  runReaderT action $ LayerReader cublas cudnn curand

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

-- elementwise log (useful for cost fct)
log :: (TensorDataType a)
    => Layer a (Trivial a)
log = fromFwdBwd fwdlog bwdlog
  where fwdlog Zero inp = return $ runST $ do
          minp <- unsafeThaw inp
          out <- MT.copy minp
          MT.tlog out
          unsafeFreeze out
        bwdlog Zero inp _ = return $ \upgrad -> (Zero, upgrad * inpgrad)
          where inpgrad = runST $ do
                  minp <- unsafeThaw inpgrad
                  out <- MT.copy minp
                  MT.inv out
                  unsafeFreeze out

-- linear layer
dot :: (TensorDataType a) => Tensor a -> Tensor a -> LayerMonad (Tensor a)
dot x y = do
  handle <- view cublasHandle
  return $ runST $ do
    let [n,m] = shape x
        [_,k] = shape y
    mx <- unsafeThaw x
    my <- unsafeThaw y
    out <- MT.emptyTensor [n,k]
    gemmFwd handle Cublas.N Cublas.N 1 mx my 0 out
    unsafeFreeze out

linear :: (TensorDataType a)
       => Layer a (Tensor a)
linear = fromFwdBwd fwdlinear bwdlinear
  where fwdlinear w x = do
          handle <- view cublasHandle
          return $ runST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            let [n,m] = shape x
                [_,k] = shape w
            out <- MT.emptyTensor [n,k]
            gemmFwd handle Cublas.N Cublas.N 1 mx mw 0 out
            unsafeFreeze out
        bwdlinear w x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            mupgrad <- unsafeThaw upgrad
            mx' <- MT.emptyTensor $ shape x
            mw' <- MT.emptyTensor $ shape w
            gemmBwdA handle Cublas.N Cublas.N 1 mw mupgrad mx'
            gemmBwdB handle Cublas.N Cublas.N 1 mx mupgrad mw'
            x' <- unsafeFreeze mx'
            w' <- unsafeFreeze mw'
            return (w', x')

-- matrix sum reductions
sumCols :: (TensorDataType a)
        => Layer a (Trivial a)
sumCols = fromFwdBwd fwdsumcols bwdsumcols
  where fwdsumcols Zero x = do
          handle <- view cublasHandle
          return $ runST $ do
            let [n,m] = shape x
            ones <- MT.fromList [1,n] $ take n $ repeat 1
            out <- MT.emptyTensor [1,m]
            mx <- unsafeThaw x
            gemmFwd handle Cublas.N Cublas.N 1 ones mx 0 out
            unsafeFreeze out
        bwdsumcols Zero x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            let [n,m] = shape x
            ones <- MT.fromList [1,n] $ take n $ repeat 1
            out <- MT.emptyTensor [n,m]
            mupgrad <- unsafeThaw upgrad
            gemmBwdB handle Cublas.N Cublas.N 1 ones mupgrad out
            outgrad <- unsafeFreeze out
            return (Zero, outgrad)

sumRows :: (TensorDataType a)
        => Layer a (Trivial a)
sumRows = fromFwdBwd fwdsumrows bwdsumrows
  where fwdsumrows Zero x = do
          handle <- view cublasHandle
          return $ runST $ do
            let [n,m] = shape x
            ones <- MT.fromList [m,1] $ take m $ repeat 1
            out <- MT.emptyTensor [n,1]
            mx <- unsafeThaw x
            gemmFwd handle Cublas.N Cublas.N 1 mx ones 0 out
            unsafeFreeze out
        bwdsumrows Zero x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            let [n,m] = shape x
            ones <- MT.fromList [m,1] $ take m $ repeat 1
            out <- MT.emptyTensor [n,m]
            mupgrad <- unsafeThaw upgrad
            gemmBwdA handle Cublas.N Cublas.N 1 ones mupgrad out
            outgrad <- unsafeFreeze out
            return (Zero, outgrad)

softmax :: (TensorDataType a) => Layer a (Trivial a)
softmax = fromFwdBwd fwdsoftmax bwdsoftmax
  where fwdsoftmax Zero x = do
          handle <- view cudnnHandle
          return $ runST $ do
            mx <- unsafeThaw x
            res <- softmaxFwd handle CuDNN.softmax_accurate
                   CuDNN.softmax_mode_instance mx
            unsafeFreeze res
        bwdsoftmax Zero x _ = do
          handle <- view cudnnHandle
          return $ \upgrad -> runST $ do
            mx <- unsafeThaw x
            mupgrad <- unsafeThaw upgrad
            xgrad <- softmaxBwd handle CuDNN.softmax_accurate
              CuDNN.softmax_mode_instance mx mupgrad >>= unsafeFreeze
            return (Zero, xgrad)

multiply :: (TensorDataType a)
         => Tensor a
         -> Layer a (Trivial a)
multiply x = fromFwdBwd fwdmul bwdmul
  where fwdmul Zero y = return $ x * y
        bwdmul Zero y _ = return $ \upgrad -> (Zero, x * upgrad)

lreshape :: forall a . (TensorDataType a)
        => [Int]
        -> Layer a (Trivial a)
lreshape shp = fromFwdBwd fwdmul bwdmul
  where fwdmul Zero x = return $ reshape shp x
        bwdmul Zero x _ = return $ \upgrad -> (Zero, reshape (shape x) upgrad)

toScalar :: (TensorDataType a) => NN LayerMonad a (Trivial a) (Tensor a) a
toScalar = NN fwdBwdToScalar
  where fwdBwdToScalar Zero x = do
          let shp = shape x
          when (product shp /= 1) $
            error $ "Can't reduce shape " ++ show shp ++ " to a scalar."
          return (head $ toList x, \upgrad -> (Zero, fromList shp [upgrad]))

-- multinomial logistic regression cost function for multi-labels
type Cost a = NN LayerMonad a (Trivial a) (Tensor a) a

mlrCost :: (TensorDataType a)
        => Tensor a
        -> Cost a
mlrCost labels =
  toScalar
  . sumCols
  . log
  . sumRows
  . multiply labels
  . lreshape (shape labels)
  . softmax
