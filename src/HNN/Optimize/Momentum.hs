{-# LANGUAGE TemplateHaskell #-}
module HNN.Optimize.Momentum (
    MomentumT
  , runMomentum
  , momentum
  ) where
import Control.Monad.RWS
import Control.Monad.Trans
import Data.VectorSpace
import Control.Lens

data MomentumState w = MomentumState {
  _previousUpdate :: w
  }
makeLenses ''MomentumState

data MomentumReader s = MomentumReader {
    _momentumFactor :: s
  , _learningRate :: s
  }
makeLenses ''MomentumReader

type MomentumT w s = RWST (MomentumReader s) () (MomentumState w)

runMomentum :: (VectorSpace w, Monad m) => Scalar w -> Scalar w -> MomentumT w (Scalar w) m a -> m a
runMomentum learningRate momentumFactor action = do
  (x,_,_) <- runRWST
             action
             (MomentumReader momentumFactor learningRate)
             (MomentumState zeroV)
  return x

momentum :: (VectorSpace w, Monad m) => Scalar w -> w -> w -> MomentumT w (Scalar w) m w
momentum cost grad w_t = do
  lr <- view learningRate
  mf <- view momentumFactor
  dtm1 <- use previousUpdate
  let dt = mf *^ dtm1 ^-^ lr *^ grad
  previousUpdate .= dt
  return $ w_t ^+^ dt
