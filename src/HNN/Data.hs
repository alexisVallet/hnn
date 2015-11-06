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
import Foreign.ForeignPtr

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

lazy_image_loader :: forall i m . (Image i, Convertible StorageImage i, Storable (ImagePixel i), MonadIO m)
                  => Proxy i -> FilePath -> Pipe (FilePath, [Int]) (Manifest (ImagePixel i), [Int]) m ()
lazy_image_loader _ directory = forever $ do
  (fpath, labels) <- await
  eimg <- {-# SCC "eimg" #-} liftIO $
          (load Autodetect $ directory </> fpath
           :: IO (Either StorageError i))
  case eimg of
   Left err -> liftIO $ do
     putStrLn $ "Unable to load image " ++ fpath
     putStrLn $ show err
   Right img -> do
     let Z:. h :. w = Friday.shape img
     if  h == 1  || w == 1
      then liftIO $ putStrLn $ "Image loaded as 1 by 1, skipping: " ++ fpath
      else do
       imgRes <- computeP img
       yield (imgRes, labels)
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

-- img_to_chw :: (Storable a, Image i, Pixel (ImagePixel i))
--            => (PixelChannel (ImagePixel i) -> a) -> i -> V.Vector a
-- img_to_chw conv img = runST $ do
--   let Z :. h :. w = Friday.shape img
--       c = nChannels img
--       imgVector = vector img
--   out <- MV.new $ c * h * w
--   forM_ [0..h-1] $ \i -> do
--     forM_ [0..w-1] $ \j -> do
--       forM_ [0..c-1] $ \k -> do
--         MV.unsafeWrite out (j + w * (i + h * k)) $ conv $ pixIndex (V.unsafeIndex imgVector (j + w * i)) k
--   V.unsafeFreeze out

-- Converts an image to a storable-based vector by casting and sharing the inner
-- data foreign pointer.
img_to_vec :: (Image i, Pixel (ImagePixel i), Storable (PixelChannel (ImagePixel i)))
           => i -> V.Vector (PixelChannel (ImagePixel i))
img_to_vec img =
  let Manifest (Z :. h :. w) pixVec = compute img
      c = nChannels img
      (pixFptr, o, l) = V.unsafeToForeignPtr pixVec
  in V.unsafeFromForeignPtr (castForeignPtr pixFptr) (c*o) (c*l)

-- applies a pixel-wise operation to all images.
map_pixels :: (FunctorImage src dst, Monad m)
           => (ImagePixel src -> ImagePixel dst) -> Pipe src dst m ()
map_pixels f = forever $ await >>= yield . Friday.map f

samesize_concat :: (Storable a) => [V.Vector a] -> V.Vector a
samesize_concat vectors = runST $ do
  let size = V.length $ head vectors
      nbVectors = length vectors
      totalSize = nbVectors * size
  mres <- MV.new totalSize
  forM_ (zip [0..] vectors) $ \(i,v) -> do
    let mresslice = MV.unsafeSlice (i * size) size mres
    mv <- V.unsafeThaw v
    MV.unsafeCopy mresslice mv
  V.unsafeFreeze mres

batch_images :: (Image i, Storable (PixelChannel (ImagePixel i)),
                 Pixel (ImagePixel i), Monad m, TensorDataType a)
             => Int
             -> Int
             -> Pipe (i, [Int]) (V.Vector (PixelChannel (ImagePixel i)), [Int], V.Vector a, [Int]) m ()
batch_images nb_labels batch_size = forever $ do
  imgAndLabels <- replicateM batch_size await
  let (images, labels) = unzip imgAndLabels
      Z :. h :. w = Friday.shape $ head images
      c = nChannels $ head images
      oneHot ls = V.fromList [if i `elem` ls then 1 else 0 | i <- [0..nb_labels - 1]]
      imgVecs = fmap img_to_vec $ images
      batch = samesize_concat imgVecs
      lmatrix = V.concat $ fmap oneHot labels
  yield (batch, [batch_size, h, w, c], lmatrix, [batch_size, nb_labels])

batch_to_gpu :: (MonadIO m, TensorDataType a) => Pipe (V.Vector a, [Int], V.Vector a, [Int]) (Tensor a, Tensor a) m ()
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
random_crop :: forall i l m . (Image i, Storable (ImagePixel i), MonadRandom m)
            => Proxy i -> Int -> Int -> Pipe (Manifest (ImagePixel i), l) (Manifest (ImagePixel i), l) m ()
random_crop _ width height = forever $ do
  (img,l) <- await
  let Z :. imgHeight :. imgWidth = Friday.shape img
  if (imgWidth < width || imgHeight < height)
   then error $ "Image too small for cropping:\nimage size: " ++ show (imgHeight,imgWidth) ++ "\ncrop size: " ++ show (height,width)
   else do
    ix <- lift $ getRandomR (0,imgWidth-width)
    iy <- lift $ getRandomR (0,imgHeight-height)
    let croppedImg = crop (Rect ix iy width height) img :: Manifest (ImagePixel i)
    cropRes <- computeP croppedImg
    yield (cropRes, l)

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
