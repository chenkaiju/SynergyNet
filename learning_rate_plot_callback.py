import tensorflow as tf
from tensorflow import keras



class LRPlot(keras.callbacks.Callback):
    # add other arguments to __init__ if you need
    def __init__(self, logger):

        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        
        lr = self.model.optimizer._decayed_lr(tf.float32)
        
        with self.logger.as_default():
            tf.summary.scalar("learning_rate", lr, step=epoch)