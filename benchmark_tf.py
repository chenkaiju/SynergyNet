import time
import argparse
import os
from math import cos, atan2, asin, sqrt, floor
import cv2
import numpy as np

from data_aflw2000_tfds import AFLW2000_TFDS
from synergynet_tf import SynergyNet
from utils.params import ParamsPack
param_pack = ParamsPack()


def parse_pose(param):
    '''parse parameters into pose'''
    if len(param)==62:
        param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    else:
        param = param * param_pack.param_std + param_pack.param_mean
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)  # yaw, pitch, roll
    return P, pose

def P2sRt(P):
    '''decomposing camera matrix P'''   
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d

def matrix2angle(R):
    '''convert matrix to angle'''
    if R[2, 0] != 1 and R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[1, 2] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[0, 1] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])
    
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi

    return [rx, ry, rz]

def parseParam(param):
    p_ = param[:, :12].reshape(-1, 3, 4)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].reshape(-1, 3, 1)
    alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
    alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
    return p, offset, alpha_shp, alpha_exp


def benchmark_aflw2000(args, checkpoint_fp, dataset):
    
    model = SynergyNet(args)
    model.load_weights(checkpoint_fp).expect_partial()

    begin = time.time()
    
    gt_true_lmk = []
    pred_lmk = []
    pred_param = []
    imgs = []
    rois = []
    
    for (img, true_landmark, roi_boxes) in dataset:

        pred_res = model.predict(img)

        pred_param.append(pred_res['pred_param'])
        pred_lmk.append(pred_res['pred_lmk'])
        gt_true_lmk.append(true_landmark)
        imgs.append(img.numpy())
        rois.append(roi_boxes)
  
    pred_param = np.vstack(pred_param)
    pred_lmk = np.vstack(pred_lmk)
    gt_true_lmk = np.vstack(gt_true_lmk)
    imgs = np.vstack(imgs)
    rois = np.vstack(rois)

    print('Extracting params take {: .3f}s'.format(time.time() - begin))
    return imgs, pred_param, pred_lmk, gt_true_lmk, rois


def calc_nme(pts68_fit_all, pts68_all_gt, roi_boxs):
    
    pts68_all = pts68_all_gt
    nme_list = []
    length_list = []
    std_size = 120
    for i in range(len(roi_boxs)):
        pts68_fit = pts68_fit_all[i]
        pts68_gt = pts68_all[i]

        sx, sy, ex, ey = roi_boxs[i]
        scale_x = (ex - sx) / std_size
        scale_y = (ey - sy) / std_size
        pts68_fit[0, :] = pts68_fit[0, :] * scale_x + sx
        pts68_fit[1, :] = pts68_fit[1, :] * scale_y + sy

        # build bbox
        minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
        miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))
        length_list.append(llength)

        dis = pts68_fit[:2, :] - pts68_gt[:2, :]
        dis = np.sqrt(np.sum(np.power(dis, 2), 0))
        dis = np.mean(dis)
        nme = dis / llength
        nme_list.append(nme)

    nme_list = np.array(nme_list, dtype=np.float32)
    return nme_list

# TODO
# def ana_msg(nme_list):

#     leng = nme_list.shape[0]
#     yaw_list_abs = np.abs(yaws_list)[:leng]
#     ind_yaw_1 = yaw_list_abs <= 30
#     ind_yaw_2 = np.bitwise_and(yaw_list_abs > 30, yaw_list_abs <= 60)
#     ind_yaw_3 = yaw_list_abs > 60

#     nme_1 = nme_list[ind_yaw_1]
#     nme_2 = nme_list[ind_yaw_2]
#     nme_3 = nme_list[ind_yaw_3]

#     mean_nme_1 = np.mean(nme_1) * 100
#     mean_nme_2 = np.mean(nme_2) * 100
#     mean_nme_3 = np.mean(nme_3) * 100

#     std_nme_1 = np.std(nme_1) * 100
#     std_nme_2 = np.std(nme_2) * 100
#     std_nme_3 = np.std(nme_3) * 100

#     mean_all = [mean_nme_1, mean_nme_2, mean_nme_3]
#     mean = np.mean(mean_all)
#     std = np.std(mean_all)

#     s0 = '\nFacial Alignment on AFLW2000-3D (NME):'
#     s1 = '[ 0, 30]\tMean: {:.3f}, Std: {:.3f}'.format(mean_nme_1, std_nme_1)
#     s2 = '[30, 60]\tMean: {:.3f}, Std: {:.3f}'.format(mean_nme_2, std_nme_2)
#     s3 = '[60, 90]\tMean: {:.3f}, Std: {:.3f}'.format(mean_nme_3, std_nme_3)
#     s4 = '[ 0, 90]\tMean: {:.3f}, Std: {:.3f}'.format(mean, std)

#     s = '\n'.join([s0, s1, s2, s3, s4])

#     return  s

# AFLW2000 facial alignment
def benchmark_aflw2000_lmk(test_imgs, lmk, gt_lmk, rois):
    '''Reconstruct the landmark points and calculate the statistics'''
    outputs = []

    batch_size = 50
    num_samples = lmk.shape[0]
    iter_num = floor(num_samples / batch_size)
    residual = num_samples % batch_size
    for i in range(iter_num+1):
        if i == iter_num:
            if residual == 0:
                break
            lm = lmk[i*batch_size: i*batch_size + residual]
            for j in range(residual):
                outputs.append(lm[j, :2, :])
        else:
            lm = lmk[i*batch_size: (i+1)*batch_size]

            for j in range(batch_size):
                if i == 0:
                    #plot the first 50 samples for validation
                    bkg = test_imgs[j]#cv2.imread(img_list[i*batch_size+j],-1)
                    bkg = cv2.cvtColor(bkg, cv2.COLOR_RGB2BGR)
                    lm_sample = lm[j]
                    c0 = np.clip((lm_sample[1,:]).astype(np.int32), 0, 119)
                    c1 = np.clip((lm_sample[0,:]).astype(np.int32), 0, 119)
                    for y, x, in zip([c0,c0,c0-1,c0-1],[c1,c1-1,c1,c1-1]):
                        bkg[y, x, :] = np.array([233,193,133])
                    cv2.imwrite(f'./results/{i*batch_size+j}.png', bkg)

                outputs.append(lm[j, :2, :])
    
    nme_list = calc_nme(lmk, gt_lmk, rois)
    
    avg_nme = np.mean(nme_list)
    return avg_nme


# AFLW2000 face orientation estimation
def benchmark_FOE(params):
    """
    FOE benchmark validation. Only calculate the groundtruth of angles within [-99, 99] (following FSA-Net https://github.com/shamangary/FSA-Net)
    """

    # AFLW200 groundturh and indices for skipping, whose yaw angle lies outside [-99, 99]
    exclude_aflw2000 = './aflw2000_data/eval/ALFW2000-3D_pose_3ANG_excl.npy'
    skip_aflw2000 = './aflw2000_data/eval/ALFW2000-3D_pose_3ANG_skip.npy'

    if not os.path.isfile(exclude_aflw2000) or not os.path.isfile(skip_aflw2000):
        raise RuntimeError('Missing data')

    pose_GT = np.load(exclude_aflw2000) 
    skip_indices = np.load(skip_aflw2000)
    pose_mat = np.ones((pose_GT.shape[0],3))

    idx = 0
    for i in range(params.shape[0]):
        if i in skip_indices:
            continue
        P, angles = parse_pose(params[i])
        angles[0], angles[1], angles[2] = angles[1], angles[0], angles[2] # we decode raw-ptich-yaw order
        pose_mat[idx,:] = np.array(angles)
        idx += 1

    pose_analyis = np.mean(np.abs(pose_mat-pose_GT),axis=0) # pose GT uses [pitch-yaw-roll] order
    MAE = np.mean(pose_analyis)
    yaw = pose_analyis[1]
    pitch = pose_analyis[0]
    roll = pose_analyis[2]
    msg = 'Mean MAE = %3.3f (in deg), [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]'%(MAE, yaw, pitch, roll)
    print('\nFace orientation estimation:')
    print(msg)
    return msg

def benchmark(checkpoint_fp, args):
    '''benchmark validation pipeline'''

    def aflw2000():
        
        aflw_tfds = AFLW2000_TFDS(batch_size=128)
        test_dataset = aflw_tfds.process(augmentation=False)

        test_imgs, pred_param, pred_lmks, gt_lmk, rois = benchmark_aflw2000(
            args=args,
            checkpoint_fp=checkpoint_fp,
            dataset=test_dataset)

        info_out_fal = benchmark_aflw2000_lmk(test_imgs, pred_lmks, gt_lmk, rois)
        print("Mean landmark NME : %3.3f"%(info_out_fal))
        #info_out_foe = benchmark_FOE(params)

    aflw2000()

def main():
    parser = argparse.ArgumentParser(description='SynergyNet benchmark on AFLW2000-3D')
    parser.add_argument('-a', '--arch', default='mobilenet_v2', type=str)
    parser.add_argument('-w', '--weights', default='./saved_model/cp-0200.ckpt', type=str)
    parser.add_argument('-d', '--device', default='0', type=str)
    parser.add_argument('--img_size', default='120', type=int)
    args = parser.parse_args()
    args.device = [int(d) for d in args.device.split(',')]

    benchmark(args.weights, args)


if __name__ == '__main__':
    main()