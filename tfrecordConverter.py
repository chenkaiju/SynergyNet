import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from utilstf.ddfa import DDFADataset_raw
from main_train_tf import parse_args


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
     """Returns an int64_list from a bool / enum / int / uint."""
     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def parse_single_image(image, label):
  
    #define the dictionary -- the structure -- of our single example
    data = {
            'image' : _bytes_feature(serialize_array(image)),
            
            'img_h' : _int64_feature(image.shape[0]),
            'img_w' : _int64_feature(image.shape[1]),
            'img_depth' : _int64_feature(image.shape[2]),
            
            'param' : _bytes_feature(serialize_array(label)) #_int64_feature(label)
        }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        "image" : tf.io.FixedLenFeature([], tf.string),
        
        "img_h" : tf.io.FixedLenFeature([], tf.int64),
        "img_w" : tf.io.FixedLenFeature([], tf.int64),
        "img_depth" : tf.io.FixedLenFeature([], tf.int64),
        
        "param" : tf.io.FixedLenFeature([], tf.string)
    }
    
    content = tf.io.parse_single_example(element, data)
    
    img = content["image"]
    
    img_h = content["img_h"]
    img_w = content["img_w"]
    img_depth = content["img_depth"]
    
    param = content["param"]
    
    # get our 'feature' and reshape it appropriately
    feature = tf.io.parse_tensor(img, out_type=tf.uint8)
    feature = tf.reshape(feature, shape=[img_h, img_w, img_depth])
    
    label = tf.io.parse_tensor(param, out_type=tf.float64)
    
    return (feature, label)

def datasetToTFrecord(dataset, dir, fname="data", num_of_samples_per_file=5000):
    file_open = False
    filePath = ""
    writer = None
    file_count = 0
    sample_count = 0
    total_samples = 0
    
    for i, (current_image, current_label) in enumerate(dataset):
        if i % num_of_samples_per_file ==0:
            if file_open:
                if filePath and writer:
                    writer.close()
                    print(f"Wrote {sample_count} elements to TFRecord file: {filePath}")
                    file_open = False
            filePath = os.path.join(dir, "{}_{}.tfrecords".format(fname, file_count))
            writer = tf.io.TFRecordWriter(filePath) #create a writer that'll store our data to disk
            sample_count = 0
            file_open = True
            file_count += 1

        out = parse_single_image(image=current_image, label=current_label)
        writer.write(out.SerializeToString())
        sample_count += 1
        total_samples += 1
    
    if file_open:
        if filePath and writer:
            writer.close()
            print(f"Wrote {sample_count} elements to TFRecord file: {filePath}")
            file_open = False
    
    print("Total samples in the dataset:", total_samples)
    
    return

def write_images_to_tfr_batched(dataset, numOfBatches_per_file, filename:str="test"):
    file_open = False
    filePath = ""
    writer = None
    file_count = 0
    sample_count = 0
    
    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        if step % numOfBatches_per_file ==0:
            if file_open:
                if filePath and writer:
                    writer.close()
                    print(f"Wrote {sample_count} elements to TFRecord file: {filePath}")
                    file_open = False
            filePath = "tfrecords/{}_{}.tfrecords".format(filename, file_count)
            writer = tf.io.TFRecordWriter(filePath) #create a writer that'll store our data to disk
            sample_count = 0
            file_open = True
            file_count += 1

        for index in range(len(x_batch_train)):

            #get the data we want to write
            current_image = x_batch_train[index] 
            current_label = y_batch_train[index]

            out = parse_single_image(image=current_image, label=current_label)
            writer.write(out.SerializeToString())
            sample_count += 1
    
    if file_open:
        if filePath and writer:
            writer.close()
            print(f"Wrote {sample_count} elements to TFRecord file: {filePath}")
            file_open = False
        
    return

def tfrecordToDataset(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)
    
    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )
    
    return dataset


def testTfrecordFile():
    filename = "tfrecords/data_0.tfrecords"
    
    dataset = tfrecordToDataset(filename)
        
    plt.figure(figsize=(10, 10))
    for step, (image, param) in enumerate(dataset):
        if step > 80:
            break
        ax = plt.subplot(9, 9, step + 1)
        plt.imshow(image.numpy())
            
    plt.savefig("test.jpg")
    
        
if __name__ == '__main__':
    # testTfrecordFile()
    args = parse_args()

    dataset = DDFADataset_raw(
        root=args.root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        shuffle=False)
    
    tfrecord_dir = "tfrecords"

    if not os.path.isdir(tfrecord_dir):
        os.mkdir(tfrecord_dir)
    
    datasetToTFrecord(dataset, tfrecord_dir, "data")
    