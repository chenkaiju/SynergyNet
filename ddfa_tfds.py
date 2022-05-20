import tensorflow_datasets as tfds
import tensorflow as tf
import custom_dataset
import math

class DDFA_TFDS():
    def __init__(self):
        self.dataset = tfds.load("custom_dataset", split='train', as_supervised=True)
        self.total = self.dataset.cardinality().numpy()
        self.batch_size = 128
        
    def process(self):
        
        train_count = tf.cast(tf.math.floor(self.total*0.9), tf.int64)
        self.train_dataset = self.dataset.take(train_count)
        self.val_dataset = self.dataset.skip(train_count)

        self.train_dataset = self.train_dataset.map(_convert_type, num_parallel_calls=8)
        self.train_dataset = self.train_dataset.batch(self.batch_size, drop_remainder=True)
        self.train_dataset = self.train_dataset.map(_augmentation, num_parallel_calls=8)
        
        self.val_dataset = self.val_dataset.map(_convert_type, num_parallel_calls=8)
        self.val_dataset = self.val_dataset.batch(self.batch_size, drop_remainder=True)
    
        return self.train_dataset, self.val_dataset

    
def _convert_type(img, param):
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img, param
    
def _augmentation(img, param):
    
    augmented_image = tf.image.random_brightness(img, 0.4)
    augmented_image = tf.image.random_saturation(augmented_image, lower=0.6, upper=1.6)
    augmented_image = tf.image.random_contrast(augmented_image, lower=0.6, upper=1.6)
    augmented_image = tf.clip_by_value(augmented_image, clip_value_min=0., clip_value_max=1.)
    
    return augmented_image, param
    

if __name__=='__main__':
    data1 = DDFA_TFDS()
    train_data, val_data = data1.process()
    
    print('train count: ', train_data.cardinality().numpy())
    print('val count: ', val_data.cardinality().numpy())
    