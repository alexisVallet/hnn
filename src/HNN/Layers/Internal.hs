{-# LANGUAGE TemplateHaskell, ScopedTypeVariables, TypeFamilies #-}
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

import HNN.Tensor
import HNN.NN
import HNN.NN.Mutable
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
  cudnn <- createHandle
  cublas <- Cublas.create
  curand <- createGenerator CuRAND.rng_pseudo_default
  CuRAND.setPseudoRandomGeneratorSeed curand rngSeed
  runReaderT action $ GPUReader cublas cudnn curand

convolution2d :: (TensorDataType a)
              => CuDNN.ConvolutionFwdAlgo
              -> (Int, Int)
              -> (Int, Int)
              -> (Int, Int)
              -> Layer GPU a (Tensor a) (Tensor a) (Tensor a)
convolution2d algo padding stride upscale =
  Layer $ fromFwdBwd convfwd convbwd
  where convfwd (filters,fmaps) = do
          handle <- view cudnnHandle
          return $ runST $ do
            filters' <- unsafeThaw filters
            fmaps' <- unsafeThaw fmaps
            convres <- convolution2dFwd handle algo padding
                       stride upscale fmaps' filters'
            unsafeFreeze convres
        convbwd (filters,fmaps) _ = do
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

activation :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
           => CuDNN.ActivationMode
           -> Layer GPU a w (Tensor a) (Tensor a)
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
pooling2d :: (TensorDataType a, VectorSpace w, Scalar w ~ a)
          => CuDNN.PoolingMode
          -> (Int, Int)
          -> (Int, Int)
          -> (Int, Int)
          -> Layer GPU a w (Tensor a) (Tensor a)
pooling2d mode size padding stride =
  noWeights $ fromFwdBwd poolfwd poolbwd
  where poolfwd fmaps = do
          handle <- view cudnnHandle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            poolres <- pooling2dFwd handle mode size padding stride fmaps'
            unsafeFreeze poolres
        poolbwd inp out = do
          handle <- view cudnnHandle
          return $ \upgrad -> runST $ do
            [inp', out', upgrad'] <- forM [inp, out, upgrad] unsafeThaw
            grad <- pooling2dBwd handle mode size padding stride
                    inp' out' upgrad'
            unsafeFreeze grad

-- dropout
dropout :: (TensorDataType a, VectorSpace w, Scalar w ~ a)
        => a
        -> Layer GPU a w (Tensor a) (Tensor a)
dropout drop_proba = noWeights $ Diff fwdbwd
  -- Simple dropout algo: generate a random tensor of 0 and 1s,
  -- elementwise multiply with the same random tensor on both forward
  -- and backward pass.
  where fwdbwd inp = do
          gen <- view generator
          mask <- liftIO $ dropoutMaskIO gen drop_proba (shape inp)
          pure_mask <- unsafeFreeze mask
          return (inp * pure_mask, \upgrad -> upgrad * pure_mask)

-- elementwise log (useful for cost fct)
llog :: (TensorDataType a, VectorSpace w, Scalar w ~ a)
     => Layer GPU a w (Tensor a) (Tensor a)
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

inv :: (TensorDataType a, VectorSpace w, Scalar w ~ a)
    => Layer GPU a w (Tensor a) (Tensor a)
inv = noWeights $ fromFwdBwd fwdInv bwdInv
  where fwdInv inp = return $ runST $ do
          minp <- unsafeThaw inp
          out <- MT.copy minp
          MT.inv out
          unsafeFreeze out
        bwdInv inp invinp = return $ \upgrad -> upgrad * (-(invinp * invinp))

-- elementwise exponential
lexp :: (TensorDataType a, VectorSpace w, Scalar w ~ a)
    => Layer GPU a w (Tensor a) (Tensor a)
lexp = noWeights $ fromFwdBwd fwdExp bwdExp
  where fwdExp inp = return $ runST $ do
          minp <- unsafeThaw inp
          out <- MT.copy minp
          MT.texp out
          unsafeFreeze out
        bwdExp inp einp = return $ \upgrad -> upgrad * einp

-- linear layer
linear :: (TensorDataType a)
       => Layer GPU a (Tensor a) (Tensor a) (Tensor a)
linear = Layer $ fromFwdBwd fwdlinear bwdlinear
  where fwdlinear (w,x) = do
          handle <- view cublasHandle
          return $ runST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            let [n,m] = shape x
                [_,k] = shape w
            out <- MT.emptyTensor [n,k]
            gemmFwd handle Cublas.N Cublas.N 1 mx mw 0 out
            unsafeFreeze out
        bwdlinear (w,x) _ = do
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
sumCols :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
        => Layer GPU a w (Tensor a) (Tensor a)
sumCols = noWeights $ fromFwdBwd fwdsumcols bwdsumcols
  where fwdsumcols x = do
          handle <- view cublasHandle
          return $ runST $ do
            let [n,m] = shape x
            ones <- MT.fromList [1,n] $ take n $ repeat 1
            out <- MT.emptyTensor [1,m]
            mx <- unsafeThaw x
            gemmFwd handle Cublas.N Cublas.N 1 ones mx 0 out
            unsafeFreeze out
        bwdsumcols x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            let [n,m] = shape x
            ones <- MT.fromList [1,n] $ take n $ repeat 1
            out <- MT.emptyTensor [n,m]
            mupgrad <- unsafeThaw upgrad
            gemmBwdB handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze out

sumRows :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
        => Layer GPU a w (Tensor a) (Tensor a)
sumRows = noWeights $ fromFwdBwd fwdsumrows bwdsumrows
  where fwdsumrows x = do
          handle <- view cublasHandle
          return $ runST $ do
            let [n,m] = shape x
            ones <- MT.fromList [m,1] $ take m $ repeat 1
            out <- MT.emptyTensor [n,1]
            mx <- unsafeThaw x
            gemmFwd handle Cublas.N Cublas.N 1 mx ones 0 out
            unsafeFreeze out
        bwdsumrows x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            let [n,m] = shape x
            ones <- MT.fromList [m,1] $ take m $ repeat 1
            out <- MT.emptyTensor [n,m]
            mupgrad <- unsafeThaw upgrad
            gemmBwdA handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze out

replicateAsRows :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
                => Int
                -> Layer GPU a w (Tensor a) (Tensor a)
replicateAsRows n = noWeights $ fromFwdBwd fwdRepRows bwdRepRows
  where fwdRepRows x = do
          handle <- view cublasHandle
          return $ runST $ do
            let [m] = shape x
            ones <- MT.fromList [n,1] $ take n $ repeat 1
            out <- MT.emptyTensor [n,m]
            mx <- unsafeThaw x >>= return . MT.reshape [1,m]
            gemmFwd handle Cublas.N Cublas.N 1 ones mx 0 out
            unsafeFreeze out
        bwdRepRows x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            let [m] = shape x
            ones <- MT.fromList [n,1] $ take n $ repeat 1
            out <- MT.emptyTensor [m]
            mupgrad <- unsafeThaw upgrad
            gemmBwdB handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze out

replicateAsCols :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
                => Int
                -> Layer GPU a w (Tensor a) (Tensor a)
replicateAsCols n = noWeights $ fromFwdBwd fwdRepCols bwdRepCols
  where fwdRepCols x = do
          handle <- view cublasHandle
          return $ runST $ do
            let [m] = shape x
            ones <- MT.fromList [1,n] $ take n $ repeat 1
            out <- MT.emptyTensor [m,n]
            mx <- unsafeThaw x >>= return . MT.reshape [m,1]
            gemmFwd handle Cublas.N Cublas.N 1 mx ones 0 out
            unsafeFreeze out
        bwdRepCols x _ = do
          handle <- view cublasHandle
          return $ \upgrad -> runST $ do
            let [m] = shape x
            ones <- MT.fromList [1,n] $ take n $ repeat 1
            out <- MT.emptyTensor [m,1]
            mupgrad <- unsafeThaw upgrad
            gemmBwdA handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze $ MT.reshape [m] out


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
softmax :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
        => Int
        -> Int
        -> Layer GPU a w (Tensor a) (Tensor a)
softmax n m =
  lexp
  >>> id &&& (sumRows >>> lreshape [n] >>> replicateAsCols m >>> inv)
  >>> multiply

scale :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
      => Layer GPU a w (a, Tensor a) (Tensor a)
scale = noWeights $ fromFwdBwd fwdscale bwdscale
  where fwdscale (x,y) = return $ x *^ y
        bwdscale (x,y) _ = return $ \upgrad -> (upgrad <.> y, x *^ upgrad)

multiply :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
         => Layer GPU a w (Tensor a, Tensor a) (Tensor a)
multiply = noWeights $ fromFwdBwd fwdMul bwdMul
  where fwdMul (x1,x2) = return $ x1 * x2
        bwdMul (x1,x2) _ = return $ \upgrad -> (x2 * upgrad, x1 * upgrad)

add :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
    => Layer GPU a w (Tensor a, Tensor a) (Tensor a)
add = noWeights $ fromFwdBwd fwdadd bwdadd
  where fwdadd (x,y) = return $ x + y
        bwdadd (x,y) _ = return $ \upgrad -> (upgrad, upgrad)

lreshape :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
        => [Int]
        -> Layer GPU a w (Tensor a) (Tensor a)
lreshape shp = noWeights $ fromFwdBwd fwdmul bwdmul
  where fwdmul x = return $ reshape shp x
        bwdmul x _ = return $ \upgrad -> reshape (shape x) upgrad

toScalar :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
         => Layer GPU a w (Tensor a) a
toScalar = noWeights $ Diff fwdBwdToScalar
  where fwdBwdToScalar x = do
          let shp = shape x
          when (product shp /= 1) $
            error $ "Can't reduce shape " ++ show shp ++ " to a scalar."
          return (head $ toList x, \upgrad -> fromList shp [upgrad])

mlrCost :: (TensorDataType a, VectorSpace w, a ~ Scalar w)
        => Int
        -> Int
        -> Layer GPU a w (Tensor a, Tensor a) a
mlrCost n c =
  let add_eps = add -< fromList [1] [10E-5] in
  id *** (add_eps >>> softmax n c)
  >>> multiply
  >>> sumRows
  >>> add_eps
  >>> llog
  >>> sumCols
  >>> scale -< (-1 / fromIntegral n)
  >>> toScalar