{-# LANGUAGE BangPatterns, TypeFamilies #-}
module Main where

import Control.Category
import Prelude hiding ((.), log)
import Foreign.C
import System.FilePath
import qualified Vision.Image as F hiding (mean)
import Pipes
import qualified Pipes.Prelude as P
import Control.Monad
import Data.VectorSpace
import Data.List
import System.Mem

import HNN as HNN

mean :: (TensorDataType a) => Tensor a -> a
mean t = (sum $ toList t) / (fromIntegral $ product $ shape t)

std :: (TensorDataType a, Real a) => Tensor a -> a
std t = let m = mean t in
  sqrt $ (sum $ fmap (\x -> (x - m)**2) $ toList t) / (fromIntegral $ product $ shape t)

print_stats :: (VectorSpace w, CFloat ~ Scalar w, MonadIO m)
            => Consumer (CFloat,w) m ()
print_stats = forever $ do
  liftIO $ putStrLn "Waiting for weights..."
  (cost, w) <- await
  liftIO $ performGC
  liftIO $ do
    putStrLn $ "Current cost: " ++ show cost
  return ()

print_info :: String -> Tensor CFloat -> GPU ()
print_info name t = liftIO $ do
  putStrLn $ name ++ ": " ++ show t
  putStrLn $ name ++ ": mean " ++ show (mean t) ++ ", std " ++ show (std t)
  putStrLn $ name ++ ": min " ++ show (minimum $ toList t) ++ ", max " ++ show (maximum $ toList t)

grey_to_float :: (Integral a) => a -> CFloat
grey_to_float i = (fromIntegral i - 128) / 255

he_init :: [Int] -> CFloat -> GPU (Tensor CFloat)
he_init shp fan_in = normal 0 (1 / sqrt fan_in) shp

main :: IO ()
main = do
  putStrLn "Loading training set..."
  mnist_train <- load_mnist $ "data" </> "mnist" </> "train" :: IO [(F.Grey, Int)]
  putStrLn "Loading test set..."
  mnist_test <- load_mnist $ "data" </> "mnist" </> "test" :: IO [(F.Grey, Int)]
  putStrLn $ "Train: " ++ show (length mnist_train) ++ " samples."
  putStrLn $ "Test: " ++ show (length mnist_test) ++ " samples."
  putStrLn $ "Train sample 0: " ++ show (F.shape $ fst $ head mnist_train)
  putStrLn $ "Test sample 0: " ++ show (F.shape $ fst $ head mnist_test)
  let batch_size = 128
      input_shape = [batch_size,1,28,28]
      cost_grad w (batch, labels) = do
        let
          convLayer = convolution2d convolution_fwd_algo_implicit_gemm (1,1) (1,1) (1,1)
                      >>> activation activation_relu
                      >>> pooling2d pooling_max (2,2) (1,1) (2,2)
          fcLayer =
            lreshape [batch_size,64*4]
            >>> linear
            >>> lreshape [batch_size,10,1,1]
            >>> activation activation_relu
          nnet =
            convLayer
            >+> convLayer
            >+> convLayer
            >+> convLayer
            >+> fcLayer
            >>> lreshape [batch_size,10]
            >>> mlrCost batch_size 10 -< labels
        (cost, bwd) <- forwardBackward nnet (w,batch)
        let (w',_) = bwd 1
        return $ (cost, w')
  runGPU 42 $ do
    conv_w1 <- he_init [8,1,3,3] (3*3)
    conv_w2 <- he_init [16,8,3,3] (3*3*8)
    conv_w3 <- he_init [32,16,3,3] (3*3*16)
    conv_w4 <- he_init [64,32,3,3] (3*3*32)
    fc_w <- normal 0 (1/ sqrt (64*4)) [64*4,10]
    let init_weights = (fc_w, (conv_w4, (conv_w3, (conv_w2, conv_w1))))
    runEffect $
      randomize (fmap (\(i,l) -> (i, [l])) mnist_train) return
      >-> batch_images grey_to_float 10 batch_size
      >-> batch_to_gpu
      >-> runMomentum 0.01 0.9 (sgd momentum cost_grad init_weights)
      >-> runEvery 10 (\(c,w) -> do
                          liftIO $ putStrLn "saving..."
                          serializeTo ("data" </> "mnist" </> "model") w)
      >-> P.take 20000
      >-> print_stats
  return ()
