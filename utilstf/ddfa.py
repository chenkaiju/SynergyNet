import os.path as osp
from pathlib import Path
import numpy as np

import cv2
import argparse
from .io import _numpy_to_tensor, _load_cpu
from .params import *
import random
from PIL import Image, ImageEnhance
import types

import tensorflow as tf

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def _is_numpy_image(img):
    return isinstance(img, np.ndarray)

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_tensor_image(img):
    return isinstance(img, tf.Tensor)


def adjust_brightness(img, brightness_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    np_img = np.array(img, dtype=np.float32)
    np_img = 255 * gain * ((np_img / 255) ** gamma)
    np_img = np.uint8(np.clip(np_img, 0, 255))

    img = Image.fromarray(np_img, 'RGB').convert(input_mode)
    return img


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# class ToTensor(object):
#     def __call__(self, pic):
#         if isinstance(pic, np.ndarray):
#             img = torch.from_numpy(pic.transpose((2, 0, 1)))
#             return img.float()

#     def __repr__(self):
#         return self.__class__.__name__ + '()'

# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, tensor):
#         tensor.sub_(self.mean).div_(self.std)
#         return tensor

# class Compose_GT(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img, gt):
#         for t in self.transforms:
#             # if isinstance(t, CenterCrop):
#             #     img, gt = t(img, gt)
#             # else:
#             #     img = t(img)
#             img = t(img)
#         return img, gt

# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img):
#         for t in self.transforms:
#             img = t(img)
#         return img

# class CenterCrop(object):
#     def __init__(self, maximum, std=None, prob=0.01, mode='train'):
#         assert(maximum >= 0)
#         self.maximum = maximum
#         self.std = std
#         self.prob = prob
#         self.type_li = [1,2,3,4,5,6,7]
#         self.switcher = {
#             1: self.lup,
#             2: self.rup,
#             3: self.ldown,
#             4: self.rdown,
#             5: self.lhalf,
#             6: self.rhalf,
#             7: self.center
#         }
#         self.mode = mode


#     def get_params(self, img):
#         h = img.shape[1]
#         w = img.shape[2]
#         crop_margins = self.maximum #random.randint(0, self.maximum)
#         rand = random.random()

#         return crop_margins, h, w, rand

#     def lup(self, img, h, w):
#         new_img = torch.zeros(3, h, w)
#         new_img[:, :h//2, :w//2] = img[:, :h//2, :w//2]
#         return new_img

#     def rup(self, img, h, w):
#         new_img = torch.zeros(3, h, w)
#         new_img[:, :h//2, w//2:] = img[:, :h//2, w//2:]
#         return new_img

#     def ldown(self, img, h, w):
#         new_img = torch.zeros(3, h, w)
#         new_img[:, h//2:, :w//2] = img[:, h//2:, :w//2]
#         return new_img

#     def rdown(self, img, h, w):
#         new_img = torch.zeros(3, h, w)
#         new_img[:, :h//2, :w//2] = img[:, :h//2, :w//2]
#         return new_img

#     def lhalf(self, img, h, w):
#         new_img = torch.zeros(3, h, w)
#         new_img[:, :, :w//2] = img[:, :, :w//2]
#         return new_img

#     def rhalf(self, img, h, w):
#         new_img = torch.zeros(3, h, w)
#         new_img[:, :, w//2:] = img[:, :, w//2:]
#         return new_img

#     def center(self, img, h, w):
#         new_img = torch.zeros(3, h, w)
#         new_img[:, h//4: -h//4, w//4: -w//4] = img[:, h//4: -h//4, w//4: -w//4]
#         return new_img

#     def __call__(self, img, gt=None):
#         crop_margins, h, w, rand = self.get_params(img)
#         crop_backgnd = torch.zeros(3, h, w)
        
#         if not(_is_tensor_image(img)):
#             raise TypeError('img should be tensor. Got {}'.format(type(img)))
#         if img.ndim == 3:
#             crop_backgnd[:, crop_margins:h-1*crop_margins, crop_margins:w-1*crop_margins] = img[:, crop_margins: h-crop_margins, crop_margins: w-crop_margins]
#             # random center crop
#             if (rand < self.prob) and (self.mode=='train'):
#                 func = self.switcher.get(random.randint(1,7))
#                 crop_backgnd = func(crop_backgnd, h, w) 

#             # center crop
#             if self.mode=='test':
#                 crop_backgnd[:, crop_margins:h-1*crop_margins, crop_margins:w-1*crop_margins] = img[:, crop_margins: h-crop_margins, crop_margins: w-crop_margins]
            
#             return crop_backgnd
#         else:
#             raise RuntimeError('img should be tensor with 3 dimensions. Got {}'.format(img.ndim))


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

# # For random cropping and GT adjustment
# class DDFADataset(data.Dataset):
#     def __init__(self, root, filelists, param_fp, gt_transform=False, transform=None, **kargs):
#         self.root = root
#         self.transform = transform
#         self.lines = Path(filelists).read_text().strip().split('\n')
#         self.params = _numpy_to_tensor(_load_cpu(param_fp))
#         self.gt_transform = gt_transform
#         self.img_loader = img_loader

#     def _target_loader(self, index):
#         target_param = self.params[index]
#         target = target_param
#         return target

#     def __getitem__(self, index):
#         path = osp.join(self.root, self.lines[index])
#         img = self.img_loader(path)
#         target = self._target_loader(index)

#         if self.transform is not None:
#             if self.gt_transform:
#                 img, target = self.transform(img, target)
#             else:
#                 img = self.transform(img)
#         return img, target

#     def __len__(self):
#         return len(self.lines)


# class DDFATestDataset(data.Dataset):
#     def __init__(self, filelists, root='', transform=None):
#         self.root = root
#         self.transform = transform
#         self.lines = Path(filelists).read_text().strip().split('\n')

#     def __getitem__(self, index):
#         path = osp.join(self.root, self.lines[index])
#         img = img_loader(path)

#         if self.transform is not None:
#             img = self.transform(img)
#         return img

#     def __len__(self):
#         return len(self.lines)


# class SGD_NanHandler(torch.optim.SGD):
#     def __init__(self, params, lr=0.1, momentum=0, dampening=0,
#                  weight_decay=0, nesterov=False):
#         super(SGD_NanHandler, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

#     @torch.no_grad()
#     def step_handleNan(self, closure=None):
#         loss = None
#         flag = False
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             weight_decay = group['weight_decay']
#             momentum = group['momentum']
#             dampening = group['dampening']
#             nesterov = group['nesterov']

#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 if True in torch.isnan(p.grad):
#                     flag = True
#                     return flag, loss
#                     #continue
#                 d_p = p.grad
#                 if weight_decay != 0:
#                     d_p = d_p.add(p, alpha=weight_decay)
#                 if momentum != 0:
#                     param_state = self.state[p]
#                     if 'momentum_buffer' not in param_state:
#                         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
#                     else:
#                         buf = param_state['momentum_buffer']
#                         buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
#                     if nesterov:
#                         d_p = d_p.add(buf, alpha=momentum)
#                     else:
#                         d_p = buf

#                 p.add_(d_p, alpha=-group['lr'])

#         return flag, loss



# class ColorJitter(object):
#     def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         self.hue = hue

#     @staticmethod
#     def get_params(brightness, contrast, saturation, hue):
#         transforms = []
#         if brightness > 0:
#             brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
#             transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

#         if contrast > 0:
#             contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
#             transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

#         if saturation > 0:
#             saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
#             transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

#         if hue > 0:
#             hue_factor = np.random.uniform(-hue, hue)
#             transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

#         np.random.shuffle(transforms)
#         transform = Compose(transforms)

#         return transform

#     def __call__(self, img):
#         if not isinstance(img, (np.ndarray, np.generic) ):
#             raise TypeError('img should be ndarray. Got {}'.format(type(img)))

#         pil = Image.fromarray(img)
#         transform = self.get_params(self.brightness, self.contrast,
#                                     self.saturation, self.hue)
#         return np.array(transform(pil))


# class Lambda(object):
#     def __init__(self, lambd):
#         assert isinstance(lambd, types.LambdaType)
#         self.lambd = lambd

#     def __call__(self, img):
#         return self.lambd(img)