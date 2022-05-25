import tensorflow_datasets as tfds
import tensorflow as tf
import aflw2000_tfds

class AFLW2000_TFDS():
    def __init__(self):
        self.trainset = tfds.load("aflw2000_tfds", data_dir='./aflw2000_tfds', split='train', as_supervised=True)
        self.total = self.trainset.cardinality().numpy()
        self.batch_size = 128
        
    def process(self, augmentation=False):

        self.trainset = self.trainset.map(_convert_type, num_parallel_calls=8)
        self.trainset = self.trainset.batch(self.batch_size, drop_remainder=True)
        
        if augmentation==True:
            self.trainset = self.trainset.map(_augmentation, num_parallel_calls=8)
            
        return self.trainset

    
def _convert_type(img, param, roi_box):
    img = tf.image.convert_image_dtype(img, tf.float32)
    param = tf.cast(param, tf.float32)

    return img, param, roi_box
    
def _augmentation(img, param, roi_box):
    
    augmented_image = tf.image.random_brightness(img, 0.4)
    augmented_image = tf.image.random_saturation(augmented_image, lower=0.6, upper=1.6)
    augmented_image = tf.image.random_contrast(augmented_image, lower=0.6, upper=1.6)
    augmented_image = tf.clip_by_value(augmented_image, clip_value_min=0., clip_value_max=1.)
    
    return augmented_image, param, roi_box
    

if __name__=='__main__':
    data1 = DDFA_TFDS()
    train_data = data1.process()
    
    print('train count: ', train_data.cardinality().numpy())
    #print('val count: ', val_data.cardinality().numpy())
    