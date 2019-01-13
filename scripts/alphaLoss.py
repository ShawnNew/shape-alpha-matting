import caffe
import numpy as np
import pdb

class AlphaMattingLossLayer(caffe.Layer):
    """
    Loss used in the first state of alpha matting.
    It is a combine of alpha-loss and compositional-loss.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        params = eval(self.param_str)
        self.w_l = params["w_l"]
        self.epsilon = params["epsilon"]
        self.shape = bottom[0].data[0][0].shape

    def reshape(self, bottom, top):
        # reshape the inter-mediate data from the blobs here.
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        # reshape the data
        self.pred = bottom[0].data[:,0,:,:]
        self.mask = bottom[1].data[:,0,:,:]
        self.color_img = bottom[1].data[:,1:4,:,:]
        self.alpha = bottom[1].data[:,4,:,:]
        self.fg = bottom[1].data[:,5:8,:,:]
        self.bg = bottom[1].data[:,8:11,:,:]
        self.pred = np.reshape(self.pred, (-1, 1, self.shape[0], self.shape[1]))
        self.mask = np.reshape(self.mask, (-1, 1, self.shape[0], self.shape[1]))
        self.alpha = np.reshape(self.alpha, (-1, 1, self.shape[0], self.shape[1]))
        self.mask[self.mask == 0.] *= 0.
        self.mask[self.mask == 1.] *= 0.
        self.mask[self.mask != 0.] = 1.
        self.num_pixels = np.sum(self.mask)

    def forward(self, bottom, top):
        top[0].data[...] = self.overall_loss(self.pred)

    def backward(self, top, propagate_down, bottom):
        self.diff = self.w_l * self.diff_alpha_ + (1-self.w_l) * self.diff_comp_
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            
            bottom[0].diff[...] = sign * self.diff / len(self.pred)

    def alpha_prediction_loss(self, pred):
        # calculate alpha_prediction_loss here
        diff = (pred - self.alpha) * self.mask                         # 4*224*224
        self.diff_alpha_ = pred - self.alpha          
        return np.sum(diff**2) / \
                (self.num_pixels + self.epsilon) / 2.

    def compositional_loss(self, pred):
        # calculate compositional_loss here
        self.color_pred = pred * self.fg + (1.0 - pred) * self.bg      # element-wise multiply to get color image
        diff = (self.color_pred - self.color_img) * self.mask          # 3 channels
        self.diff_comp_ = np.average(
            (self.color_pred-self.color_img) * (self.fg-self.bg), \
            axis=1)
        self.diff_comp_ = np.reshape(self.diff_comp_, (-1, 1, self.shape[0], self.shape[1]))   
        diff = np.average(diff, axis=1)                      # average over color channel
        return np.sum(diff**2) / \
                (self.num_pixels + self.epsilon) / 2.


    def overall_loss(self, pred):                
        # average the above two losses
        alpha_loss = self.alpha_prediction_loss(pred)
        comp_loss = self.compositional_loss(pred)
        return self.w_l * alpha_loss + \
                (1 - self.w_l) * comp_loss