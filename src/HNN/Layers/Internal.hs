{-# LANGUAGE TemplateHaskell #-}
module HNN.Layers.Internal (
    GPU
  , cudnnHandle
  , cublasHandle
  , generator
  , convolution2d
  , CuDNN.convolution_fwd_algo_implicit_gemm
  , CuDNN.convolution_fwd_algo_implicit_precomp_gemm
  , CuDNN.convolution_fwd_algo_gemm
  , CuDNN.convolution_fwd_algo_direct
  , ConvOutShape
  , activation
  , CuDNN.activation_sigmoid
  , CuDNN.activation_relu
  , CuDNN.activation_tanh
  , pooling2d
  , CuDNN.pooling_max
  , CuDNN.pooling_average_count_include_padding
  , CuDNN.pooling_average_count_exclude_padding
  , PoolOutShape
  , dropout
  , linear
  , sumCols
  , sumRows
  , replicateAsRows
  , replicateAsCols  
  , runGPU
  , CuDNN.ConvolutionFwdAlgo
  , CuDNN.ActivationMode
  , CuDNN.PoolingMode
  , lreshape
  , mlrCost
  , toScalar
  , softmax
  , multiply
  , add 
  , llog
  , lexp
  , inv
  , HNN.Layers.Internal.nchw_to_nhwc
  , HNN.Layers.Internal.nhwc_to_nchw
  ) where
import Prelude hiding ((.), id)
import Foreign.C
import Foreign.Marshal
import Control.Monad.Reader
import Control.Monad.RWS
import Control.Monad.ST
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.CuRAND as CuRAND
import Control.Lens hiding ((<.>))
import Control.Category
import Control.Monad.Primitive
import System.IO.Unsafe
import Data.HList.HList
import Data.Proxy
import GHC.TypeLits

import HNN.Tensor
import HNN.NN
import HNN.NN.Mutable as M
import qualified HNN.Tensor.Mutable as MT

data GPUReader = GPUReader {
  _cublasHandle :: Cublas.Handle,
  _cudnnHandle :: CuDNN.Handle,
  _generator :: CuRAND.Generator
  }
makeLenses ''GPUReader

type GPU  = ReaderT GPUReader IO

-- Actually running the thing.
runGPU :: CULLong -> GPU a -> IO a
runGPU rngSeed action = do
  putStrLn "Initi cublas..."
  cublas <- Cublas.create
  putStrLn "Init cudnn..."
  cudnn <- createHandle
  putStrLn "Init curand..."
  curand <- createGenerator CuRAND.rng_pseudo_default
  putStrLn "Setting PRNG seed..."
  CuRAND.setPseudoRandomGeneratorSeed curand rngSeed
  putStrLn "Running action..."
  runReaderT action $ GPUReader cublas cudnn curand

convolution2d :: (TensorDataType a, Shape input_shape, Shape filter_shape,
                  Shape padding, Shape stride, Shape out_shape,
                  out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                  Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4, Nbdim out_shape ~ 4,
                  Nbdim stride ~ 2, Nbdim padding ~ 2)
              => Proxy [padding,stride]
              -> CuDNN.ConvolutionFwdAlgo
              -> Layer GPU a
                   '[Tensor filter_shape a]
                   (Tensor input_shape a)
                   (Tensor out_shape a)
convolution2d p algo =
  Layer $ fromFwdBwd convfwd convbwd
  where convfwd (HLS (HCons filters HNil),fmaps) = do
          handle <- view cudnnHandle
          return $ runST $ do
            filters' <- unsafeThaw filters
            fmaps' <- unsafeThaw fmaps
            convres <- convolution2dFwd p handle algo fmaps' filters'
            unsafeFreeze convres
        convbwd (HLS (HCons filters HNil),fmaps) _ = do
          handle <- view cudnnHandle
          let bwdfilters upgrad = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                filtersgrad <- convolution2dBwdFilters p handle fmaps' filters' upgrad'
                unsafeFreeze filtersgrad
              bwdinputs upgrad = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                inputsgrad <- convolution2dBwdInputs p handle fmaps' filters' upgrad'
                unsafeFreeze inputsgrad
          return $ \upgrad -> (HLS $ bwdfilters upgrad `HCons` HNil, bwdinputs upgrad)

activation :: (TensorDataType a, Shape s, Nbdim s ~ 4)
           => CuDNN.ActivationMode
           -> Layer GPU a '[] (Tensor s a) (Tensor s a)
activation mode =
  noWeights $ fromFwdBwd actfwd actbwd
  where actfwd fmaps = do
          handle <- view cudnnHandle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            activations <- activationFwd handle mode fmaps'
            unsafeFreeze activations
        actbwd inp out = do
          handle <- view cudnnHandle
          return $ \upgrad -> runST $ do
            inp' <- unsafeThaw inp
            out' <- unsafeThaw out
            upgrad' <- unsafeThaw upgrad
            grad <- activationBwd handle mode inp' out' upgrad'
            unsafeFreeze grad

-- pooling
pooling2d :: (TensorDataType a, Shape input_shape, Shape pooling_size,
              Shape padding, Shape stride, Shape out_shape,
              out_shape ~ PoolOutShape input_shape pooling_size padding stride,
              Nbdim input_shape ~ 4, Nbdim out_shape ~ 4, Nbdim pooling_size ~ 2,
              Nbdim padding ~ 2, Nbdim stride ~ 2)
          => Proxy [pooling_size,padding,stride]
          -> CuDNN.PoolingMode
          -> Layer GPU a '[] (Tensor input_shape a) (Tensor out_shape a)
pooling2d p mode =
  noWeights $ fromFwdBwd poolfwd poolbwd
  where poolfwd fmaps = do
          handle <- view cudnnHandle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            poolres <- pooling2dFwd p handle mode fmaps'
            unsafeFreeze poolres
        poolbwd inp out = do
          handle <- view cudnnHandle
          return $ \upgrad -> runST $ do
            inp' <- unsafeThaw inp
            out' <- unsafeThaw out
            upgrad' <- unsafeThaw upgrad
            grad <- pooling2dBwd p handle mode inp' out' upgrad'
            unsafeFreeze grad

-- dropout
dropout :: (TensorDataType a, Shape s)
        => a
        -> Layer GPU a '[] (Tensor s a) (Tensor s a)
dropout drop_proba = noWeights $ Diff fwdbwd
  -- Simple dropout algo: generate a random tensor of 0 and 1s,
  -- elementwise multiply with the same random tensor on both forward
  -- and backward pass.
  where fwdbwd inp = do
          gen <- view generator
          mask <- liftIO $ dropoutMaskIO gen drop_proba
          pure_mask <- unsafeFreeze mask
          return (inp * pure_mask, \upgrad -> upgrad * pure_mask)

-- elementwise log (useful for cost fct)
llog :: (TensorDataType a, Shape s)
     => Layer GPU a '[] (Tensor s a) (Tensor s a)
llog = noWeights $ fromFwdBwd fwdlog bwdlog
  where fwdlog inp = return $ runST $ do
          minp <- unsafeThaw inp
          out <- MT.copy minp
          MT.tlog out
          unsafeFreeze out
        bwdlog inp _ = return $ \upgrad -> upgrad * inpgrad
          where inpgrad = runST $ do
                  minp <- unsafeThaw inp
                  out <- MT.copy minp
                  MT.inv out
                  unsafeFreeze out

inv :: (TensorDataType a, Shape s)
    => Layer GPU a '[] (Tensor s a) (Tensor s a)
inv = noWeights $ fromFwdBwd fwdInv bwdInv
  where fwdInv inp = return $ runST $ do
          minp <- unsafeThaw inp
          out <- MT.copy minp
          MT.inv out
          unsafeFreeze out
        bwdInv inp invinp = return $ \upgrad -> upgrad * (-(invinp * invinp))

-- elementwise exponential
lexp :: (TensorDataType a, Shape s)
     => Layer GPU a '[] (Tensor s a) (Tensor s a)
lexp = noWeights $ fromFwdBwd fwdExp bwdExp
  where fwdExp inp = return $ runST $ do
          minp <- unsafeThaw inp
          out <- MT.copy minp
          MT.texp out
          unsafeFreeze out
        bwdExp inp einp = return $ \upgrad -> upgrad * einp

-- linear layer
linear :: (TensorDataType a, KnownNat m, KnownNat n, KnownNat k)
       => Layer GPU a '[Tensor [m,k] a] (Tensor [n,m] a) (Tensor [n,k] a)
linear = Layer $ fromFwdBwd fwdlinear bwdlinear
  where fwdlinear (HLS (HCons w HNil),x) = do
          handle <- view cublasHandle
          return $ runST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            out <- MT.emptyTensor
            gemmFwd handle Cublas.N Cublas.N 1 mx mw 0 out
            unsafeFreeze out
        bwdlinear (HLS (HCons w HNil),x) _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            mupgrad <- unsafeThaw upgrad
            mx' <- MT.emptyTensor
            mw' <- MT.emptyTensor
            gemmBwdA handle Cublas.N Cublas.N 1 mw mupgrad mx'
            gemmBwdB handle Cublas.N Cublas.N 1 mx mupgrad mw'
            x' <- unsafeFreeze mx'
            w' <- unsafeFreeze mw'
            return (HLS $ w' `HCons` HNil, x')

-- matrix sum reductions
sumCols :: forall a n m . (TensorDataType a, KnownNat n, KnownNat m)
        => Layer GPU a '[] (Tensor [n,m] a) (Tensor [1,m] a)
sumCols = noWeights $ fromFwdBwd fwdsumcols bwdsumcols
  where fwdsumcols x = do
          handle <- view cublasHandle
          return $ runST $ do
            ones <- MT.onesP (Proxy :: Proxy [1,n])
            out <- MT.emptyTensor
            mx <- unsafeThaw x
            gemmFwd handle Cublas.N Cublas.N 1 ones mx 0 out
            unsafeFreeze out
        bwdsumcols x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            ones <- MT.onesP (Proxy :: Proxy [1,n])
            out <- MT.emptyTensor
            mupgrad <- unsafeThaw upgrad
            gemmBwdB handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze out

sumRows :: forall a n m . (TensorDataType a, KnownNat n, KnownNat m)
        => Layer GPU a '[] (Tensor [n,m] a) (Tensor [n,1] a)
sumRows = noWeights $ fromFwdBwd fwdsumrows bwdsumrows
  where fwdsumrows x = do
          handle <- view cublasHandle
          return $ runST $ do
            ones <- MT.onesP (Proxy :: Proxy [m,1])
            out <- MT.emptyTensor
            mx <- unsafeThaw x
            gemmFwd handle Cublas.N Cublas.N 1 mx ones 0 out
            unsafeFreeze out
        bwdsumrows x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            ones <- MT.onesP (Proxy :: Proxy [m,1])
            out <- MT.emptyTensor
            mupgrad <- unsafeThaw upgrad
            gemmBwdA handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze out

replicateAsRows :: forall a n m
                . (TensorDataType a, KnownNat n, KnownNat m)
                => Proxy n
                -> Layer GPU a '[] (Tensor '[m] a) (Tensor [n,m] a)
replicateAsRows _ = noWeights $ fromFwdBwd fwdRepRows bwdRepRows
  where fwdRepRows x = do
          handle <- view cublasHandle
          return $ runST $ do
            ones <- MT.onesP (Proxy :: Proxy [n,1])
            out <- MT.emptyTensor
            mx <- unsafeThaw x >>= return . MT.reshapeP (Proxy :: Proxy [1,m])
            gemmFwd handle Cublas.N Cublas.N 1 ones mx 0 out
            unsafeFreeze out
        bwdRepRows x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            ones <- MT.onesP (Proxy :: Proxy [n,1])
            out <- MT.emptyTensorP (Proxy :: Proxy '[1,m])
            mupgrad <- unsafeThaw upgrad
            gemmBwdB handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze $ MT.reshape out

replicateAsCols :: forall a n m
                . (TensorDataType a, KnownNat n, KnownNat m)
                => Proxy n
                -> Layer GPU a '[] (Tensor '[m] a) (Tensor [m,n] a)
replicateAsCols _ = noWeights $ fromFwdBwd fwdRepCols bwdRepCols
  where fwdRepCols x = do
          handle <- view cublasHandle
          return $ runST $ do
            ones <- MT.onesP (Proxy :: Proxy [1,n])
            out <- MT.emptyTensor
            mx <- unsafeThaw x >>= return . MT.reshapeP (Proxy :: Proxy [m,1])
            gemmFwd handle Cublas.N Cublas.N 1 mx ones 0 out
            unsafeFreeze out
        bwdRepCols x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            ones <- MT.onesP (Proxy :: Proxy [1,n])
            out <- MT.emptyTensorP (Proxy :: Proxy [m,1])
            mupgrad <- unsafeThaw upgrad
            gemmBwdA handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze $ MT.reshape out

-- softmax :: (TensorDataType a) => GPU a (Trivial a)
-- softmax = fromFwdBwd fwdsoftmax bwdsoftmax
--   where fwdsoftmax Zero x = do
--           handle <- view cudnnHandle
--           return $ runST $ do
--             mx <- unsafeThaw x
--             res <- softmaxFwd handle CuDNN.softmax_accurate
--                    CuDNN.softmax_mode_instance mx
--             unsafeFreeze res
--         bwdsoftmax Zero x _ = do
--           handle <- view cudnnHandle
--           return $ \upgrad -> runST $ do
--             mx <- unsafeThaw x
--             mupgrad <- unsafeThaw upgrad
--             xgrad <- softmaxBwd handle CuDNN.softmax_accurate
--               CuDNN.softmax_mode_instance mx mupgrad >>= unsafeFreeze
--             return (Zero, xgrad)

-- "Naive" GPU implementation of softmax, as workaround to bug in current
-- CuDNN-based softmax.
softmax :: (TensorDataType a, KnownNat n, KnownNat m)
        => Layer GPU a '[] (Tensor [n,m] a) (Tensor [n,m] a)
softmax =
  lexp
  >+> id &&& (sumRows >+> lreshape >+> replicateAsCols (Proxy :: Proxy m) >+> inv)
  >+> multiply

scale :: (TensorDataType a, Shape s)
      => Layer GPU a '[] (a, Tensor s a) (Tensor s a)
scale = noWeights $ fromFwdBwd fwdscale bwdscale
  where fwdscale (x,y) = return $ x *^ y
        bwdscale (x,y) _ = return $ \upgrad -> (upgrad <.> y, x *^ upgrad)

multiply :: (TensorDataType a, Shape s)
         => Layer GPU a '[] (Tensor s a, Tensor s a) (Tensor s a)
multiply = noWeights $ fromFwdBwd fwdMul bwdMul
  where fwdMul (x1,x2) = return $ x1 * x2
        bwdMul (x1,x2) _ = return $ \upgrad -> (x2 * upgrad, x1 * upgrad)

add :: (TensorDataType a, Shape s)
    => Layer GPU a '[] (Tensor s a, Tensor s a) (Tensor s a)
add = noWeights $ fromFwdBwd fwdadd bwdadd
  where fwdadd (x,y) = return $ x + y
        bwdadd (x,y) _ = return $ \upgrad -> (upgrad, upgrad)

lreshape :: (TensorDataType a, Shape s1, Shape s2, Size s1 ~ Size s2)
         => Layer GPU a '[] (Tensor s1 a) (Tensor s2 a)
lreshape = noWeights $ fromFwdBwd fwdmul bwdmul
  where fwdmul x = return $ reshape x
        bwdmul _ _ = return $ \upgrad -> reshape upgrad

toScalar :: (TensorDataType a, Shape s, Size s ~ 1)
         => Layer GPU a '[] (Tensor s a) a
toScalar = noWeights $ Diff fwdBwdToScalar
  where fwdBwdToScalar x = do
          return (head $ toList x, \upgrad -> fromList [upgrad])

mlrCost :: forall a n m . (TensorDataType a, KnownNat n, KnownNat m)
        => Layer GPU a '[] (Tensor [n,m] a, Tensor [n,m] a) a
mlrCost =
  id *** (add -< 10E-5 >+> softmax)
  >+> multiply
  >+> sumRows
  >+> add -< 10E-5
  >+> llog
  >+> sumCols
  >+> scale -< (-1 / fromIntegral (natVal (Proxy :: Proxy n)))
  >+> toScalar

nchw_to_nhwc :: forall n c h w a
                . (TensorDataType a, KnownNat n, KnownNat c, KnownNat h, KnownNat w)
                => Layer GPU a '[] (Tensor [n,c,h,w] a) (Tensor [n,h,w,c] a)
nchw_to_nhwc = noWeights $ fromFwdBwd fwdTrans bwdTrans
  where fwdTrans t = do
          handle <- view cudnnHandle
          return $ runST $ do
            mt <- unsafeThaw t
            mt' <- M.nchw_to_nhwc handle mt
            unsafeFreeze mt'
        bwdTrans _ _ = do
          handle <- view cudnnHandle
          return $ \upgrad -> runST $ do
            mu <- unsafeThaw upgrad
            mu' <- M.nhwc_to_nchw handle mu
            unsafeFreeze mu'

nhwc_to_nchw :: forall n c h w a
                . (TensorDataType a, KnownNat n, KnownNat c, KnownNat h, KnownNat w)
                => Layer GPU a '[] (Tensor [n,h,w,c] a) (Tensor [n,c,h,w] a)
nhwc_to_nchw = noWeights $ fromFwdBwd fwdTrans bwdTrans
  where fwdTrans t = do
          handle <- view cudnnHandle
          return $ runST $ do
            mt <- unsafeThaw t
            mt' <- M.nhwc_to_nchw handle mt
            unsafeFreeze mt'
        bwdTrans _ _ = do
          handle <- view cudnnHandle
          return $ \upgrad -> runST $ do
            mu <- unsafeThaw upgrad
            mu' <- M.nchw_to_nhwc handle mu
            unsafeFreeze mu'
