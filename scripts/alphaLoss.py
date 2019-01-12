import caffe
import numpy as np
import glog
import pdb

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
        self.pred = bottom[0].data[:,0,:,:]
        self.mask = bottom[1].data[:,0,:,:]
        self.color_img = bottom[1].data[:,1:4,:,:]
        self.alpha = bottom[1].data[:,4,:,:]
        self.fg = bottom[1].data[:,5:8,:,:]
        self.bg = bottom[1].data[:,8:11,:,:]
        self.pred = np.reshape(self.pred, (-1, 1, self.shape[0], self.shape[1]))
        self.mask = np.reshape(self.mask, (-1, 1, self.shape[0], self.shape[1]))
        self.alpha = np.reshape(self.alpha, (-1, 1, self.shape[0], self.shape[1]))
        top[0].reshape(1)

    def forward(self, bottom, top):
        # calculate loss here
        top[0].data[...] = self.overall_loss(
                                             self.pred, 
                                             self.mask,
                                             self.alpha,
                                             self.color_img,
                                             self.fg,
                                             self.bg)        


    def backward(self, top, progagate_down, bottom):
        # pass gradient back
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~BackProp~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        bottom[0].diff[...] = self.computeGradient(
                                                   self.pred,
                                                   self.mask,
                                                   self.alpha,
                                                   self.color_img,
                                                   self.fg,
                                                   self.bg)

    def alpha_prediction_loss(self, mask, pred, alpha):
        # calculate alpha_prediction_loss here
        diff = (pred - alpha) * mask                         # 4*224*224          
        num_pixels = np.sum(mask)
        sqrt_ = np.sqrt(np.square(diff) + self.epsilon**2)
        return np.sum(sqrt_) \
        / (num_pixels + self.epsilon)

    def compositional_loss(self, pred, mask, color_img, fg, bg):
        # calculate compositional_loss here
        mask = np.reshape(mask, (-1, 1, self.shape[0], self.shape[1]))
        pred = np.reshape(pred, (-1, 1, self.shape[0], self.shape[1]))
        self.color_pred = pred * fg + (1.0 - pred) * bg      # element-wise multiply to get color image
        diff = (self.color_pred - color_img) * mask          # 3 channels      
        num_pixels = np.sum(mask)
        diff = np.average(diff, axis=1)                      # average over color channel
        sqrt_ = np.sqrt(np.square(diff) + self.epsilon**2)   # shape of (4*1*224*224)
        return np.sum(sqrt_) \
        / (num_pixels + self.epsilon)


    def overall_loss(self, pred, mask, alpha, color_img, fg, bg):                
        # average the above two losses
        mask[mask == 0.] *= 0.
        mask[mask == 1.] *= 0.
        mask[mask != 0.] = 1.                                # extract mask area
        alpha_loss = self.alpha_prediction_loss(mask, pred, alpha)
        comp_loss = self.compositional_loss(pred, mask, color_img, fg, bg)
        return self.w_l * alpha_loss + \
                (1 - self.w_l) * comp_loss

    def computeGradient(self, pred, mask, alpha, color_img, fg, bg):
        # calculate gradient here
        alpha_diff_ = pred - alpha
        alpha_sqrt_ = np.sqrt(np.square(alpha_diff_) + self.epsilon**2)
        comp_diff_ = self.color_pred - color_img
        comp_sqrt_ = np.sqrt(np.square(comp_diff_) + self.epsilon**2)
        fb_diff_ = fg - bg  # difference between foreground and background
        
        alpha_gradient_ = 2 * alpha_diff_ / alpha_sqrt_
        comp_gradient_ = 2 * comp_diff_ * fb_diff_ / comp_sqrt_
        comp_gradient_ = np.average(comp_gradient_, axis=1) # average over color channel
        comp_gradient_ = np.reshape(comp_gradient_, (-1, 1, self.shape[0], self.shape[1]))

        overall_gradient_ = self.w_l * alpha_gradient_ + \
                         (1 - self.w_l) * comp_gradient_    # over gradient along pixel
        print overall_gradient_
        return overall_gradient_
          
