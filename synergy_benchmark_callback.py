import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import io

from utilstf.params import ParamsPack
param_pack = ParamsPack()

class BenchmarkCallback(keras.callbacks.Callback):
    def __init__(self, data, params, roi_boxes, logger):
        self.test_images = data
        self.test_landmarks = params
        self.test_roiBoxes = roi_boxes
        
        self.test_landmarks_rescaled = \
            np.array([self.rescaleLmk(landmark.numpy(), roi_box.numpy()) for landmark, roi_box in zip(params, roi_boxes)])
        
        self.logger = logger
        
    def on_epoch_end(self, epoch, logs=None):
        
        result_test = self.model(self.test_images)
        params_pred_test = result_test['pred_param'].numpy()
        
        height, width = self.test_images[0].shape[:2]
        base = 3
        
        target_roi_box = [0, 0, 120, 120, 1]
        
        paramToLmk = lambda x: self.predict_sparseVert(x, target_roi_box, transform=True)
        lmks_pred = np.array([paramToLmk(param) for param in params_pred_test])
        
        # Calculate lmk loss
        res = lmks_pred[:,:] - self.test_landmarks_rescaled[:,:]
        # l2_avg_loss = np.average(np.sqrt(np.sum(res**2, axis=1)))
        l1_avg_loss = np.average(np.abs(res))
        
        # Visualize data
        figure, ax_array = plt.subplots(2, 3, figsize=(height/width*base*3, base*2))
        figure.suptitle('Green=True, Red=Predict (aflw2000)')#, fontsize=16
        
        for i, ax in zip(range(6), np.ravel(ax_array)):
            single_img = self.test_images[i].numpy().astype(np.uint8)
            single_landmark = self.test_landmarks_rescaled[i]
            
            lmk_pred = lmks_pred[i]
    
            self.draw_landmarks(single_img, single_landmark, ax, color='g', markeredgecolor='green')
            self.draw_landmarks(single_img, lmk_pred, ax, color='r', markeredgecolor='red')
    
        landmark_img = self.plot_to_image(figure) 
        with self.logger.as_default():
            tf.summary.image("Benchmark data", landmark_img, step=epoch, max_outputs=6)
            tf.summary.scalar("Benchmark average l1 loss", (l1_avg_loss), step=epoch)
    
    def rescaleLmk(self, vertex, roi_bbox, target_roi_bbox=[0, 0, 120, 120]):
        sx, sy, ex, ey = roi_bbox
        scale_x = (target_roi_bbox[2] - target_roi_bbox[0]) / (ex - sx)
        scale_y = (target_roi_bbox[3] - target_roi_bbox[1]) / (ey - sy)
        vertex[0, :] = vertex[0, :] - sx
        vertex[1, :] = vertex[1, :] - sy
        vertex[0, :] = vertex[0, :] * scale_x
        vertex[1, :] = vertex[1, :] * scale_y

        s = (scale_x + scale_y) / 2
        vertex[2, :] *= s

        return vertex

    def predict_sparseVert(self, param, roi_box, transform=False):
        return self._predict_vertices(param, roi_box, dense=False, transform=transform)

    def _predict_vertices(self, param, roi_bbox, dense, transform=True):
        vertex = self.param2vert(param, dense=dense, transform=transform)
        sx, sy, ex, ey, _ = roi_bbox
        scale_x = (ex - sx) / 120
        scale_y = (ey - sy) / 120
        vertex[0, :] = vertex[0, :] * scale_x + sx
        vertex[1, :] = vertex[1, :] * scale_y + sy

        s = (scale_x + scale_y) / 2
        vertex[2, :] *= s

        return vertex
    
    def param2vert(self, param, dense=False, transform=True):
        if param.shape[0] == 62:
            param_ = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
        else:
            raise RuntimeError('length of params mismatch')

        p, offset, alpha_shp, alpha_exp = self.parse_param(param_)

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
    
    def parse_param(self, param):
        param = param
        p_ = param[:12].reshape(3, 4)
        p = p_[:, :3]
        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = param[12:52].reshape(40, 1)
        alpha_exp = param[52:62].reshape(10, 1)
        return p, offset, alpha_shp, alpha_exp
        
    def draw_landmarks(self, img, pts, fig, color='g', markeredgecolor = 'green'):
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
    
    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=3)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image  