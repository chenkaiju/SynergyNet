import numpy as np
import cv2
import time

from utilstf.inference import crop_img, predict_sparseVert, draw_landmarks, predict_denseVert, predict_pose, draw_axis
import argparse
import os
import os.path as osp
import glob
from FaceBoxes import FaceBoxes
from utils.render import render
import scipy.io as sio
import onnxruntime as onnxrt

from utilstf.params import ParamsPack
param_pack = ParamsPack()

IMG_SIZE = 120

def main(args):
    
    load_start = time.process_time()
    session = onnxrt.InferenceSession("./pred_model/onnx/saved_model.onnx",
                                      providers=['TensorrtExecutionProvider', 
                                                 'CUDAExecutionProvider', 
                                                 'CPUExecutionProvider'])
    load_end = time.process_time()
    print("load time: {}".format(load_end-load_start))
    print(onnxrt.get_device())
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

    avg = 0
    count = 0  
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
            
           
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input = img_rgb.astype(np.float32)[np.newaxis, ...]
            
            # inferences
            inference_start = time.process_time()
            inp_dct = {'input': input}
            param_pred = session.run(None, inp_dct)[0]
            inference_end = time.process_time()
            avg += (inference_end-inference_start)
            count += 1
            print("inference time: {}".format(inference_end-inference_start))
            param_pred = param_pred.squeeze(axis=0)
            cv2.imwrite(f'inference_output/validate_crop/validate_{idx}.png', img)
            
            
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

    avg /= count
    print("avg inference time: ", avg)   
 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default='./faces/', help='path to a single image or path to a folder containing multiple images')
    parser.add_argument("--png", action="store_true", help="if images are with .png extension")
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int)

    args = parser.parse_args()
    main(args)