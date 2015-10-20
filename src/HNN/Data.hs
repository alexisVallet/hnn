{-# LANGUAGE FlexibleContexts, ScopedTypeVariables, TypeFamilies #-}
module HNN.Data (
    load_mnist
  , randomize
  , map_pixels
  , batch_images_labels
  , normalize
  , runEvery
  , serializeTo
  ) where

import Foreign.C
import System.IO
import System.FilePath
import Vision.Image as Friday hiding (read)
import Vision.Primitive
import Vision.Image.Storage.DevIL
import Pipes
import System.Directory
import Data.Maybe
import Control.Monad
import Control.Monad.ST
import Control.Monad.Random
import Foreign.Storable
import qualified Data.Vector.Storable as V
import qualified Data.Vector as NSV
import qualified Data.Vector.Mutable as NSVM
import qualified Data.ByteString as B
import Data.Serialize

import HNN.Tensor as T

load_mnist :: (Convertible StorageImage i)
           => FilePath -> IO [(i, Int)]
load_mnist directory = do
  dirs <- getDirectoryContents directory >>= filterM (\d -> doesDirectoryExist $ directory </> d)
  imgs <- forM dirs $ \classname -> do
    imgfpaths <- getDirectoryContents (directory </> classname) 
    imgs' <- forM imgfpaths $ \imgfpath -> do
        eimg <-load Autodetect $ directory </> classname </> imgfpath
        case eimg of
          Left err -> return Nothing
          Right img -> return $ Just (img, read classname)
    return imgs'
  return $ fmap fromJust $ filter (\mi -> case mi of
                                      Nothing -> False
                                      _ -> True) $ concat $ imgs

shuffle :: MonadRandom m => [a] -> m [a]
shuffle xs = do
  let l = length xs
  rands_ <- getRandomRs (0, l-1)
  let
    rands = take l rands_
    ar = runST $ do
      ar <- NSV.unsafeThaw $ NSV.fromList xs
      forM_ (zip [0..(l-1)] rands) $ \(i, j) -> do
        vi <- NSVM.read ar i
        vj <- NSVM.read ar j
        NSVM.write ar j vi
        NSVM.write ar i vj
      NSV.unsafeFreeze ar
  return $ NSV.toList ar

randomize :: MonadRandom m => [a] -> (a -> m b) -> Producer b m ()
randomize xs f = do
  sxs <- lift $ shuffle xs
  forM_ sxs $ \x -> do
    fx <- lift $ f x
    yield fx

-- converts an image to a (nb_channels, height, width) shaped row-major
-- vector.
img_to_chw :: (Storable a, Image i, Pixel (ImagePixel i))
           => (PixelChannel (ImagePixel i) -> a) -> i -> V.Vector a
img_to_chw conv img = V.fromList [conv $ pixIndex (index img (ix2 i j)) k
                             | k <- [0..c-1],
                               i <- [0..h-1],
                               j <- [0..w-1]]
  where Z :. h :. w = Friday.shape img
        c = nChannels img

-- applies a pixel-wise operation to all images.
map_pixels :: (FunctorImage src dst, Monad m)
           => (ImagePixel src -> ImagePixel dst) -> Pipe src dst m ()
map_pixels f = forever $ await >>= yield . Friday.map f

-- batch images with the same shape.
batch_images_labels :: forall i m a .
                       (Image i, TensorDataType a, Pixel (ImagePixel i), Monad m)
                    => (PixelChannel (ImagePixel i) -> a) -> Int -> Int -> Pipe (i, [Int]) (Tensor a, Tensor a) m ()
batch_images_labels conv nb_labels batch_size = forever $ do
  imgAndLabels <- replicateM batch_size await
  let (images, labels) = unzip imgAndLabels
      Z :. h :. w = Friday.shape $ head images
      c = nChannels $ head images
      oneHot ls = V.fromList [if i `elem` ls then 1 else 0 | i <- [0..nb_labels - 1]]
      batch = fromVector [batch_size, c, h, w] $ V.concat $ fmap (img_to_chw conv) images
      lmatrix = fromVector [batch_size, nb_labels] $ V.concat $ fmap oneHot labels
  yield (batch, lmatrix)

-- normalizes batches with given mean and standard deviation
normalize :: (Monad m, TensorDataType a) => a -> a -> Pipe (Tensor a, Tensor a) (Tensor a, Tensor a) m ()
normalize mean std = forever $ do
  (batch, labels) <- await
  yield $ ((1/std) *^ batch - T.fromList [1] [mean], labels)

-- serializes inputs
runEvery :: (Monad m) => Int -> (a -> m ()) -> Pipe a a m c
runEvery n action = forever $ do
  replicateM (n - 1) $ await >>= yield
  x <- await
  lift $ action x
  yield x

serializeTo :: (MonadIO m, Serialize a) => FilePath -> a -> m ()
serializeTo fpath toSerialize = do
  liftIO $ B.writeFile fpath $ encode toSerialize
