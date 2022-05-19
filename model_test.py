import os
from re import I
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import random
import numpy as np
import glob

import tensorflow_datasets as tfds
import custom_dataset

from synergynet_train import parse_args
from synergynet_tf import SynergyNet as SynergyNet
from utilstf.ddfa import DDFADataset
from utilstf.params import ParamsPack
param_pack = ParamsPack()

def parse_param(param):
    param = param.numpy()
    p_ = param[:12].reshape(3, 4)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(40, 1)
    alpha_exp = param[52:62].reshape(10, 1)
    return p, offset, alpha_shp, alpha_exp

def param2vert(param, dense=False, transform=True):
    if param.shape[0] == 62:
        param_ = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    else:
        raise RuntimeError('length of params mismatch')

    p, offset, alpha_shp, alpha_exp = parse_param(param_)

    if dense:
        vertex = p @ (param_pack.u + param_pack.w_shp @ alpha_shp + param_pack.w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
        if transform: 
            # transform to image coordinate space
            vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]
    else:
        vertex = p @ (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset
        if transform: 
            # transform to image coordinate space
            vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]

    return vertex

def _predict_vertices(param, roi_bbox, dense, transform=True):
    vertex = param2vert(param, dense=dense, transform=transform)
    sx, sy, ex, ey, _ = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s

    return vertex

def predict_sparseVert(param, roi_box, transform=False):
    return _predict_vertices(param, roi_box, dense=False, transform=transform)

def draw_landmarks_plt(img, pts, fig, color='g', markeredgecolor = 'green'):

    fig.imshow(img[:, :])
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.axis('off')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        alpha = 0.8
        markersize = 1.5
        lw = 1 

        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

        # close eyes and mouths
        plot_close = lambda i1, i2: fig.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                color=color, lw=lw, alpha=alpha - 0.1)
        plot_close(41, 36)
        plot_close(47, 42)
        plot_close(59, 48)
        plot_close(67, 60)

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            fig.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

            fig.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                        color=color,
                        markeredgecolor=markeredgecolor, alpha=alpha)

    return fig

def drawLandmark(image, pts, color='g', markeredgecolor = 'green'):
    
    if not type(pts) in [tuple, list]:
        pts = [pts]
        
    for i in range(len(pts)):
        alpha = 0.8
        markersize = 1.5
        lw = 1 

        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

        # close eyes and mouths
        #cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
        plot_close = lambda i1, i2: cv2.line(image, (int(pts[i][0, i1]), int(pts[i][1, i1])), 
                                                    (int(pts[i][0, i2]), int(pts[i][1, i2])), color=(0, 0, 1), thickness=lw)
        plot_close(41, 36)
        plot_close(47, 42)
        plot_close(59, 48)
        plot_close(67, 60)

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            # fig.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)
            for i in range(l, r-1):
                plot_close(i, i+1)

            # fig.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
            #             color=color,
            #             markeredgecolor=markeredgecolor, alpha=alpha)
        
    image = image*255
    image = image.astype(np.int)

    return image

def predictFace(imgPath, count, dir="./mod2", ckpt_name='cp-0065.ckpt'):
    
    size = 512
    fname = os.path.basename(imgPath)
    # imgPath = "./img/female1.jpg"
    
    image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = image / 255.0
    
    args = parse_args()
    
    model = SynergyNet(args)

    resume_model = os.path.join(dir, ckpt_name)
    model.load_weights(resume_model)
    
    img_pred = cv2.resize(image, (120, 120), interpolation=cv2.INTER_LINEAR)
    pred = model(np.array([img_pred]), training=False)['pred_param'][0]
    
    roi_box = [0, 0, size, size, 1]
    
    lmks_pred = predict_sparseVert(pred, roi_box, transform=True)
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    draw_landmarks_plt(image, lmks_pred, ax, color='g', markeredgecolor='green')
    
    filePath = os.path.join(dir, fname)
    plt.savefig(filePath)
    print("Done for {}".format(count))
    
    return


def main():
    args = parse_args()
    
    model = SynergyNet(args)

    resume_model = os.path.join('./mod2', 'cp-0065.ckpt')
    # ckpt_dir = "./mod"
    model.load_weights(resume_model)
    
    
    # prepare dataset
    _, _, test_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        batch_size= args.batch_size,
        gt_transform=True,
        transform=[]
    )
    roi_box = [0, 0, 120, 120, 1]
    for iter, (imgs, params) in enumerate(test_dataset):
        
        preds = model(imgs, training=False)['pred_param']
        
        fig = plt.figure(figsize=(10, 10))
        fig.set_tight_layout(True)
        for i in range(4):
            
            rand = random.randint(0, args.batch_size-1)
            image, param = imgs[rand].numpy(), params[rand]
            pred = preds[rand]
                                    
            lmks_truth = predict_sparseVert(param, roi_box, transform=True)
            lmks_pred = predict_sparseVert(pred, roi_box, transform=True)
            
            
            ax = plt.subplot(2, 4, i*2 + 1)
            draw_landmarks_plt(image, lmks_truth, ax, color='g', markeredgecolor='green')
            ax = plt.subplot(2, 4, i*2 + 2)
            draw_landmarks_plt(image, lmks_pred, ax, color='r', markeredgecolor='red')
            
        fname = "./mod/test_{}.jpg".format(iter)
        plt.savefig(fname)
        print(fname)
        
    
    return


if __name__ == '__main__':
    
    # ds = tfds.load('custom_dataset')
    
    # for i, sample in enumerate(ds['train']):
    #     img = sample['image']
    #     param = sample['param']
    #     img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
    #     cv2.imwrite("test.jpg", img)
    #     print("")
    
    dataDir = "./mod2/aflw2000_data/AFLW2000-3D_crop"
    paths = glob.glob(os.path.join(dataDir, "*.jpg"))
    for i, path in enumerate(paths):
        # if i < 10:
        predictFace(path, I)
    # main()