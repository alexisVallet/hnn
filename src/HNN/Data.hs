{-# LANGUAGE FlexibleContexts, ScopedTypeVariables, TypeFamilies #-}
module HNN.Data (
    load_mnist
  , lazy_image_loader  
  , randomize
  , map_pixels
  , batch_images
  , batch_to_gpu
  , normalize
  , runEvery
  , serializeTo
  , random_crop
  , timePipe
  ) where

import Foreign.C
import System.IO
import System.FilePath
import Vision.Image as Friday hiding (read)
import Vision.Primitive
import Vision.Image.Storage.DevIL
import Pipes hiding (Proxy)
import System.Directory
import Data.Maybe
import Control.Monad
import Control.Monad.ST
import Control.Monad.Random
import Foreign.Storable
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV
import qualified Data.Vector as NSV
import qualified Data.Vector.Mutable as NSVM
import qualified Data.ByteString as B
import Data.Serialize
import Data.Proxy
import qualified Data.Time as Time
import Data.IORef
import Control.DeepSeq

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

lazy_image_loader :: (Convertible StorageImage i, MonadIO m)
                  => Proxy i -> FilePath -> Pipe (FilePath, [Int]) (i, [Int]) m ()
lazy_image_loader _ directory = forever $ do
  (fpath, labels) <- await
  eimg <- liftIO $ load Autodetect $ directory </> fpath
  case eimg of
   Left err -> liftIO $ do
     putStrLn $ "Unable to load image " ++ fpath
     putStrLn $ show err
   Right img -> do
     yield (img, labels)
     return ()

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
-- img_to_chw :: (Storable a, Image i, Pixel (ImagePixel i))
--            => (PixelChannel (ImagePixel i) -> a) -> i -> V.Vector a
-- img_to_chw conv img = V.fromList [conv $ pixIndex (index img (ix2 i j)) k
--                              | k <- [0..c-1],
--                                i <- [0..h-1],
--                                j <- [0..w-1]]
--   where Z :. h :. w = Friday.shape img
--         c = nChannels img

img_to_chw :: (Storable a, Image i, Pixel (ImagePixel i))
           => (PixelChannel (ImagePixel i) -> a) -> i -> V.Vector a
img_to_chw conv img = runST $ do
  let Z :. h :. w = Friday.shape img
      c = nChannels img
      imgVector = vector img
  out <- MV.new $ c * h * w
  forM_ [0..h-1] $ \i -> do
    forM_ [0..w-1] $ \j -> do
      forM_ [0..c-1] $ \k -> do
        MV.unsafeWrite out (j + w * (i + h * k)) $ conv $ pixIndex (V.unsafeIndex imgVector (j + w * i)) k
  V.unsafeFreeze out

-- applies a pixel-wise operation to all images.
map_pixels :: (FunctorImage src dst, Monad m)
           => (ImagePixel src -> ImagePixel dst) -> Pipe src dst m ()
map_pixels f = forever $ await >>= yield . Friday.map f

batch_images :: (Image i, TensorDataType a, Pixel (ImagePixel i), Monad m)
             => (PixelChannel (ImagePixel i) -> a) -> Int -> Int -> Pipe (i, [Int]) (V.Vector a, [Int], V.Vector a, [Int]) m ()
batch_images conv nb_labels batch_size = forever $ do
  imgAndLabels <- replicateM batch_size await
  let (images, labels) = unzip imgAndLabels
      Z :. h :. w = Friday.shape $ head images
      c = nChannels $ head images
      oneHot ls = V.fromList [if i `elem` ls then 1 else 0 | i <- [0..nb_labels - 1]]
      batch = V.concat $ fmap (img_to_chw conv) images
      lmatrix = V.concat $ fmap oneHot labels
  yield (batch, [batch_size, c, h, w], lmatrix, [batch_size, nb_labels])

batch_to_gpu :: (Monad m, TensorDataType a) => Pipe (V.Vector a, [Int], V.Vector a, [Int]) (Tensor a, Tensor a) m ()
batch_to_gpu = forever $ do
  (batch, bshape, labels, lshape) <- await
  yield (fromVector bshape batch, fromVector lshape labels)

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

-- Dataset transformations
random_crop :: (MonadRandom m, Image i1, FromFunction i1, ImagePixel i1 ~ FromFunctionPixel i1)
            => Int -> Int -> Pipe (i1,l) (i1,l) m ()
random_crop width height = forever $ do
  (img,l) <- await
  let Z :. imgHeight :. imgWidth = Friday.shape img
  if (imgWidth < width || imgHeight < height)
   then error $ "Image too small for cropping:\nimage size: " ++ show (imgHeight,imgWidth) ++ "\ncrop size: " ++ show (height,width)
   else do
    ix <- lift $ getRandomR (0,imgWidth-width)
    iy <- lift $ getRandomR (0,imgHeight-height)
    let croppedImg = crop (Rect ix iy width height) img
    yield (croppedImg, l)

timePipe :: (NFData a, NFData b, MonadIO m, MonadIO m')
         => String -> Pipe a b m c -> m' (Pipe a b m c)
timePipe name pab = liftIO $ do
  startRef <- Time.getCurrentTime >>= newIORef
  let setStartPipe = forever $ do
        a <- await
        deepseq a $ liftIO $ Time.getCurrentTime >>= writeIORef startRef
        yield a
      printTimePipe = forever $ do
        b <- await
        deepseq b $ liftIO $ do
          end_time <- Time.getCurrentTime
          start_time <- readIORef startRef
          putStrLn $ name ++ " ran in " ++ show (Time.diffUTCTime end_time start_time)
        yield b
  return $ setStartPipe >-> pab >-> printTimePipe
