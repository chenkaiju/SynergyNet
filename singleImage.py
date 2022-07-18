from cv2 import transform
import numpy as np
import tensorflow as tf
import cv2
from synergynet_tf import SynergyNet
from utilstf.inference import crop_img, predict_sparseVert, draw_landmarks, predict_denseVert, predict_pose, draw_axis
import argparse
import os
import os.path as osp
import glob
from FaceBoxes import FaceBoxes
from utils.render import render
import scipy.io as sio

from utilstf.params import ParamsPack
param_pack = ParamsPack()

# Following 3DDFA-V2, we also use 120x120 resolution
IMG_SIZE = 120

    
def main(args):
    
    args.arch = 'mobilenet_v2'
    model = SynergyNet(args)
    
    # load pre-tained model
    dir="./saved_model"
    ckpt_name='cp-0200.ckpt'
    resume_model = os.path.join(dir, ckpt_name)
    #resume_model = os.path.join('./ckpts_tfds', 'cp-0061.ckpt')
    model.load_weights(resume_model).expect_partial()

    # face detector
    face_boxes = FaceBoxes()

    if osp.isdir(args.files):
        if not args.files[-1] == '/':
            args.files = args.files + '/'
        if not args.png:
            files = sorted(glob.glob(args.files+'*.jpg'))
        else:
            files = sorted(glob.glob(args.files+'*.png'))
    else:
        files = [args.files]

    for img_fp in files:
        print("Process the image: ", img_fp)

        img_ori = cv2.imread(img_fp)

        # crop faces
        rects = face_boxes(img_ori)

        # storage
        pts_res = []
        poses = []
        vertices_lst = []
        if not osp.exists(f'inference_output/validate_crop/'):
            os.makedirs(f'inference_output/validate_crop/')
            
        for idx, rect in enumerate(rects):
            roi_box = rect

            # enlarge the bbox a little and do a square crop
            HCenter = (rect[1] + rect[3])/2
            WCenter = (rect[0] + rect[2])/2
            side_len = roi_box[3]-roi_box[1]
            margin = side_len * 1.2 // 2
            roi_box[0], roi_box[1], roi_box[2], roi_box[3] = WCenter-margin, HCenter-margin, WCenter+margin, HCenter+margin

            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            
            cv2.imwrite(f'inference_output/validate_crop/validate_{idx}.png', img)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input = tf.cast(img_rgb, tf.float32)
            input = tf.expand_dims(input, 0)
            
            res = model(input, training=False)
            param_pred_batch = res['pred_param']
            
            param_pred = tf.squeeze(param_pred_batch, [0]).numpy()
            
            # inferences
            lmks = predict_sparseVert(param_pred, roi_box, transform=True)
            vertices = predict_denseVert(param_pred, roi_box, transform=True)
            angles, translation = predict_pose(param_pred, roi_box)

            pts_res.append(lmks)
            vertices_lst.append(vertices)
            poses.append([angles, translation, lmks])

        if not osp.exists(f'inference_output/rendering_overlay/'):
            os.makedirs(f'inference_output/rendering_overlay/')
        if not osp.exists(f'inference_output/landmarks/'):
            os.makedirs(f'inference_output/landmarks/')
        if not osp.exists(f'inference_output/poses/'):
            os.makedirs(f'inference_output/poses/')
        
        name = img_fp.rsplit('/',1)[-1][:-4]
        img_ori_copy = img_ori.copy()

        # mesh
        tri = sio.loadmat('./3dmm_data/tri.mat')['tri'] - 1
        render(img_ori, vertices_lst, alpha=0.6, wfp=f'inference_output/rendering_overlay/{name}.jpg', 
               connectivity=tri)
        
        # landmarks
        draw_landmarks(img_ori_copy, pts_res, wfp=f'inference_output/landmarks/{name}.jpg')
        
        # face orientation
        img_axis_plot = img_ori_copy
        for angles, translation, lmks in poses:
            img_axis_plot = draw_axis(img_axis_plot, angles[0], angles[1],
                angles[2], translation[0], translation[1], size = 50, pts68=lmks)
        wfp = f'inference_output/poses/{name}.jpg'
        cv2.imwrite(wfp, img_axis_plot)
        print(f'Save pose result to {wfp}')

# def transformToROI(vertex, roi_bbox):
    
#     sx, sy, ex, ey, _ = roi_bbox
#     scale_x = (ex - sx) / 120
#     scale_y = (ey - sy) / 120
#     vertex[0, :] = vertex[0, :] * scale_x + sx
#     vertex[1, :] = vertex[1, :] * scale_y + sy

#     s = (scale_x + scale_y) / 2
#     vertex[2, :] *= s
    
#     return vertex
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default='./faces/', help='path to a single image or path to a folder containing multiple images')
    parser.add_argument("--png", action="store_true", help="if images are with .png extension")
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int)

    args = parser.parse_args()
    main(args)