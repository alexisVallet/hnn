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
import System.Directory
import qualified Data.Vector.Storable as V
import qualified Database.LevelDB as LDB
import Control.Monad.Trans.Resource

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
  -- Attempts to load the leveldb for mnist.
  -- Populates it if doesn't exist.
  let train_ldb_path = "data" </> "mnist" </> "train_ldb"
      test_ldb_path = "data" </> "mnist" </> "test_ldb"
      train_img_path = "data" </> "mnist" </> "train"
      test_img_path = "data" </> "mnist" </> "test"
  putStrLn "Building training data leveldb if not existing..."
  runResourceT $ runEffect
    $ (load_mnist_lazy train_img_path :: Producer (F.Grey, Int) (ResourceT IO) ())
    >-> makeleveldb train_ldb_path Nothing
  putStrLn "Building test data leveldb if not existing..."
  runResourceT $ runEffect
    $ (load_mnist_lazy test_img_path :: Producer (F.Grey, Int) (ResourceT IO) ())
    >-> makeleveldb test_img_path Nothing
  mnist_train <- runResourceT $ LDB.open train_ldb_path LDB.defaultOptions
  mnist_test <- runResourceT $ LDB.open test_ldb_path LDB.defaultOptions
  let batch_size = 128
      convLayer = convolution2d convolution_fwd_algo_implicit_gemm (1,1) (1,1) (1,1)
                  >+> activation activation_relu
                  >+> pooling2d pooling_max (2,2) (1,1) (2,2)
      fcLayer = lreshape [batch_size,64*4]
                >+> linear
                >+> lreshape [batch_size,10,1,1]
                >+> activation activation_relu
      nnet =
        transformTensor nhwc nchw
        >+> convLayer
        >+> convLayer
        >+> convLayer
        >+> convLayer
        >+> fcLayer
        >+> lreshape [batch_size,10]
      input_shape = [batch_size,1,28,28]
      cost_grad w (batch, labels) = do
        let fullNet = nnet >+> mlrCost batch_size 10 -< labels
        (cost, bwd) <- lift $ forwardBackward fullNet (w,batch)
        let (w',_) = bwd 1
        return $ (cost, w')
  runGPU 42 $ do
    conv_w1 <- he_init [8,1,3,3] (3*3)
    conv_w2 <- he_init [16,8,3,3] (3*3*8)
    conv_w3 <- he_init [32,16,3,3] (3*3*16)
    conv_w4 <- he_init [64,32,3,3] (3*3*32)
    fc_w <- normal 0 (1/ sqrt (64*4)) [64*4,10]
    let init_weights =
          HLS $ conv_w1 `HCons` conv_w2 `HCons` conv_w3 `HCons` conv_w4 `HCons` fc_w `HCons` HNil
    runResourceT $ runEffect $
      (leveldb_random_loader mnist_train :: Producer (F.Grey, [Int]) (ResourceT GPU) ())
      >-> batch_images 10 batch_size
      >-> P.map (\(b,bs,l,ls) -> (V.map (\p -> (fromIntegral p - 128) / 255 :: CFloat) b,
                                  bs,l,ls))
      >-> batch_to_gpu
      >-> runMomentum 0.01 0.9 (sgd momentum cost_grad init_weights)
      >-> runEvery 10 (\(c,w) -> do
                          liftIO $ putStrLn "saving..."
                          serializeTo ("data" </> "mnist" </> "model") w)
      >-> P.take 20000
      >-> print_stats
  return ()
