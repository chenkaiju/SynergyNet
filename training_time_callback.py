import tensorflow as tf
import time

class TrainTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger):
        self.logger = logger
        
    def on_train_begin(self, logs):
        self.train_start = time.perf_counter()
    
    def on_epoch_begin(self, epoch, logs):
        self.epoch_start = time.perf_counter()
        
    def on_epoch_end(self, epoch, logs):
        with self.logger.as_default():
            tf.summary.scalar("epoch_time", (time.perf_counter()-self.epoch_start), step=epoch)
            tf.summary.scalar("training_time", (time.perf_counter()-self.train_start), step = epoch)
        