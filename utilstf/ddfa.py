import os.path as osp
from pathlib import Path
import numpy as np

import cv2
import argparse
from .io import _load_cpu
from .params import *
from PIL import Image, ImageEnhance
import types

import tensorflow as tf

# def img_loader(path):
#     return cv2.imread(path, cv2.IMREAD_COLOR)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

# def _is_numpy_image(img):
#     return isinstance(img, np.ndarray)

# def _is_pil_image(img):
#     return isinstance(img, Image.Image)

# def _is_tensor_image(img):
#     return isinstance(img, tf.Tensor)


# def adjust_brightness(img, brightness_factor):
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     enhancer = ImageEnhance.Brightness(img)
#     img = enhancer.enhance(brightness_factor)
#     return img


# def adjust_contrast(img, contrast_factor):
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     enhancer = ImageEnhance.Contrast(img)
#     img = enhancer.enhance(contrast_factor)
#     return img


# def adjust_saturation(img, saturation_factor):
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     enhancer = ImageEnhance.Color(img)
#     img = enhancer.enhance(saturation_factor)
#     return img


# def adjust_hue(img, hue_factor):
#     if not(-0.5 <= hue_factor <= 0.5):
#         raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     input_mode = img.mode
#     if input_mode in {'L', '1', 'I', 'F'}:
#         return img

#     h, s, v = img.convert('HSV').split()

#     np_h = np.array(h, dtype=np.uint8)
#     # uint8 addition take cares of rotation across boundaries
#     with np.errstate(over='ignore'):
#         np_h += np.uint8(hue_factor * 255)
#     h = Image.fromarray(np_h, 'L')

#     img = Image.merge('HSV', (h, s, v)).convert(input_mode)
#     return img


# def adjust_gamma(img, gamma, gain=1):
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     if gamma < 0:
#         raise ValueError('Gamma should be a non-negative real number')

#     input_mode = img.mode
#     img = img.convert('RGB')

#     np_img = np.array(img, dtype=np.float32)
#     np_img = 255 * gain * ((np_img / 255) ** gamma)
#     np_img = np.uint8(np.clip(np_img, 0, 255))

#     img = Image.fromarray(np_img, 'RGB').convert(input_mode)
#     return img


# class AverageMeter(object):
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


def DDFADataset(root, filelists, param_fp, batch_size=8, gt_transform=False, transform=None):
    
    lines = Path(filelists).read_text().strip().split('\n')
    img_path = [osp.join(root,s) for s in lines]
    params = _load_cpu(param_fp)[:,:62] #12 pose, 40 shape, 10 expression, 40 texture
    
    total_data = len(lines)
    split_train = int(total_data*0.9)
    split_validation = int(total_data*0.95)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((img_path[:split_train], params[:split_train]))
    val_dataset = tf.data.Dataset.from_tensor_slices((img_path[split_train:split_validation], params[split_train:split_validation]))
    test_dataset = tf.data.Dataset.from_tensor_slices((img_path[split_validation:], params[split_validation:]))
    
    train_dataset = train_dataset.shuffle(buffer_size=split_train)
    train_dataset = train_dataset.map(_process_pathnames, num_parallel_calls=8)
    train_dataset = train_dataset.map(_augmentation, num_parallel_calls=8)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    
    val_dataset = val_dataset.map(_process_pathnames, num_parallel_calls=8)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    
    test_dataset = test_dataset.map(_process_pathnames, num_parallel_calls=8)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    
    return train_dataset, val_dataset, test_dataset

def DDFADataset_raw(root, filelists, param_fp, shuffle=False):
    
    lines = Path(filelists).read_text().strip().split('\n')
    img_path = [osp.join(root,s) for s in lines]
    params = _load_cpu(param_fp)[:,:62] #12 pose, 40 shape, 10 expression, 40 texture
    
    dataset = tf.data.Dataset.from_tensor_slices((img_path, params))
    if shuffle:
        total_data = len(lines)
        dataset = dataset.shuffle(buffer_size=total_data)
    dataset = dataset.map(_read_imgfiles, num_parallel_calls=8)
    
    return dataset
        
def _augmentation(image, param):

    augmented_image = tf.image.random_brightness(image, 0.4)
    #augmented_image = tf.image.random_hue(augmented_image, 0.02)
    augmented_image = tf.image.random_saturation(augmented_image, lower=0.6, upper=1.6)
    augmented_image = tf.image.random_contrast(augmented_image, lower=0.6, upper=1.6)
    augmented_image = tf.clip_by_value(augmented_image, clip_value_min=0., clip_value_max=1.)
    
    return augmented_image, param
    
def _process_pathnames(image_path, param):
    # We map this function onto each pathname pair
    img_str = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_str, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img, param

def _read_imgfiles(image_path, param):
    img_str = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_str, channels=3)
    return img, param