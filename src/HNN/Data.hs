{-# LANGUAGE FlexibleContexts, ScopedTypeVariables, TypeFamilies #-}
module HNN.Data where

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

import HNN.Tensor as T

load_mnist :: (Convertible StorageImage i)
           => FilePath -> IO [(i, Int)]
load_mnist directory = do
  dirs <- getDirectoryContents directory >>= filterM doesDirectoryExist
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
img_to_chw :: (Image i, Pixel (ImagePixel i), Storable (PixelChannel (ImagePixel i)))
           => i -> V.Vector (PixelChannel (ImagePixel i))
img_to_chw img = V.fromList [pixIndex (index img (ix2 i j)) k
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
batch_images :: forall i m a .
                (Image i, TensorDataType a, Pixel (ImagePixel i),
                 PixelChannel (ImagePixel i) ~ a, Monad m)
             => Int -> Pipe i (Tensor a) m ()
batch_images batch_size = forever $ do
  images <- replicateM batch_size $ await
  let Z :. h :. w = Friday.shape $ head images
      c = nChannels $ head images
  yield $ fromVector [batch_size, c, h, w] $ V.concat $ fmap img_to_chw images
