import caffe
import numpy as np
import glog
import cv2
import pdb
from PIL import Image

class AlphaMattingLossLayer(caffe.Layer):
    """
    Add Root Square Loss Layer.

    Use like this:

    layer {
        name: "xxx"
        type: "Python"
        bottom: "pred"
        bottom: "label"
        top: "loss"
        include {
            phase: train/test
        }
        python_param {
            module: "rootSquareLoss"
            layer: "RootSquareLossLayer"
            param_str: "xxx"
        }
    }
    """
    # Check the input blob, it should have two input and the dimension should align
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception(
                "Wrong number of bottom blobs \
                (prediction and ground truth)")
        # if bottom[0].data.ndim != bottom[1].data.ndim:
        #     raise Exception("Prediction's dimension does not align with the label")
        params = eval(self.param_str)
        self.epsilon = params["epsilon"]
        self.w_l = params["w_l"]
        self.shape = bottom[0].data[0][0].shape
        glog.info('loss layer setup done.')

    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):
        # calculate loss here
        pred = bottom[0].data[:, 0, :, :]
        mask = bottom[1].data[:, 0, :, :]
        color_img = bottom[1].data[:, 1:4, :, :]
        alpha = bottom[1].data[:, 4, :, :]
        fg = bottom[1].data[:, 5:8, :, :]
        bg = bottom[1].data[:, 8:11, :, :]
        top[0].data[...] = self.overall_loss(pred, mask, alpha, color_img, fg, bg)        


    def backward(self, top, progagate_down, bottom):
        self.diff_alpha = np.reshape(self.diff_alpha, (-1, 1, self.shape[0], self.shape[1])) 
        self.sqrt_alpha = np.reshape(self.sqrt_alpha, (-1, 1, self.shape[0], self.shape[1]))
        self.diff_comp = np.reshape(self.diff_alpha, (-1, 1, self.shape[0], self.shape[1])) 
        self.sqrt_comp = np.reshape(self.sqrt_comp, (-1, 1, self.shape[0], self.shape[1]))       
        temp = self.w_l * self.diff_alpha / self.sqrt_alpha + \
                                (1 - self.w_l) * self.diff_comp / self.sqrt_comp
        bottom[0].diff[...] = temp
       
    def alpha_prediction_loss(self, mask, pred, alpha):
        # calculate alpha_prediction_loss here
        diff = pred - alpha                                  # 4*224*224          
        diff = diff * mask                                   # element-wise multiply        
        self.diff_alpha = diff
        num_pixels = np.sum(mask)
        self.sqrt_alpha = np.sqrt(np.square(self.diff_alpha) + self.epsilon**2)
        return np.sum(self.sqrt_alpha) \
        / (num_pixels + self.epsilon)

    def compositional_loss(self, pred, mask, color_img, fg, bg):
        # calculate compositional_loss here
        mask = np.reshape(mask, (-1, 1, self.shape[0], self.shape[1]))
        pred = np.reshape(pred, (-1, 1, self.shape[0], self.shape[1]))
        color_pred = pred * fg + (1.0 - pred) * bg           # element-wise multiply to get color image
        diff = color_pred - color_img                        # 3 channels      
        diff = diff * mask
        self.diff_comp = np.average(diff, axis=1)            # average over rgb
        num_pixels = np.sum(mask)
        self.sqrt_comp = np.sqrt(
            np.square(self.diff_comp) + self.epsilon**2)     # shape of (4*1*224*224)
        return np.sum(self.sqrt_comp) \
        / (num_pixels + self.epsilon)


    def overall_loss(self, pred, mask, alpha, color_img, fg, bg):                
        # average the above two losses        
        # mask[mask == 0.] *= 0.
        # mask[mask == 1.] *= 0.
        # mask[mask != 0.] = 1.
        alpha_loss = self.alpha_prediction_loss(mask, pred, alpha)
        comp_loss = self.compositional_loss(pred, mask, color_img, fg, bg)
        return self.w_l * alpha_loss + \
                (1 - self.w_l) * comp_loss
        
