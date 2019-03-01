import caffe
import numpy as np
import utils
from utils import generate_gradient_map
import pdb

class AlphaPredictionLossLayer(caffe.Layer):
    """
    Loss used in the first state of alpha matting.
    It is a combine of alpha-loss and compositional-loss.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        params = eval(self.param_str)
        self.epsilon = params["epsilon"]
        self.shape = bottom[0].data[0][0].shape

    def reshape(self, bottom, top):
        # reshape the inter-mediate data from the blobs here.
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        # reshape the data
        self.pred = bottom[0].data[:,0,:,:]
        mask_ = bottom[1].data[:,0,:,:] / 255.
        self.alpha = bottom[1].data[:,1,:,:] / 255.
        self.pred = np.reshape(self.pred, (-1, 1, self.shape[0], self.shape[1]))
        mask_ = np.reshape(mask_, (-1, 1, self.shape[0], self.shape[1]))
        self.alpha = np.reshape(self.alpha, (-1, 1, self.shape[0], self.shape[1]))
        mask_[mask_ == 0.] *= 0.
        mask_[mask_ == 1.] *= 0.
        mask_[mask_ != 0.] = 1.
        self.mask = mask_
        self.num_pixels = np.sum(self.mask)

    def forward(self, bottom, top):
        #print np.sum(self.pred)
        top[0].data[...] = self.alpha_prediction_loss(self.pred)

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            
            bottom[i].diff[...] = sign * self.grad_alpha

    def alpha_prediction_loss(self, pred):
        # calculate alpha_prediction_loss here
        diff_ = pred - self.alpha
        # loss
        loss_ = np.sum((diff_**2) * self.mask) /\
                 (2 * (self.num_pixels + self.epsilon))
        self.diff_alpha = diff_

        # compute gradient
        self.grad_alpha = diff_ * self.mask / len(self.pred)
        return loss_



class AlphaPredictionWithGradientLossLayer(caffe.Layer):
    """
    Loss used in the first state of alpha matting.
    It is a combine of alpha-loss and compositional-loss.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        params = eval(self.param_str)
        self.epsilon = params["epsilon"]
        self.shape = bottom[0].data[0][0].shape
        w_ = {}
        w_["w_a"] = params["w_a"]
        w_["w_g"] = params["w_g"]
        self.weight = w_

    def reshape(self, bottom, top):
        # reshape the inter-mediate data from the blobs here.
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        # reshape the data
        self.pred = bottom[0].data[:,0,:,:]
        mask_ = bottom[1].data[:,0,:,:] / 255.
        self.alpha = bottom[1].data[:,1,:,:] / 255.
        gradient = bottom[1].data[:,2,:,:] / 255.
        self.pred = np.reshape(self.pred, (-1, 1, self.shape[0], self.shape[1]))
        mask_ = np.reshape(mask_, (-1, 1, self.shape[0], self.shape[1]))
        self.alpha = np.reshape(self.alpha, (-1, 1, self.shape[0], self.shape[1]))
        mask_[mask_ == 0.] *= 0.
        mask_[mask_ == 1.] *= 0.
        mask_[mask_ != 0.] = 1.
        self.mask = mask_
        gradient = np.reshape(gradient, (-1, 1, self.shape[0], self.shape[1]))
        self.gradient_map = generate_gradient_map(gradient, 3)
        self.num_pixels = np.sum(self.mask)

    def forward(self, bottom, top):
        #print np.sum(self.pred)
        top[0].data[...] = self.weight["w_a"] * self.alpha_prediction_loss(self.pred) +\
                           self.weight["w_g"] * self.gradient_loss()

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            
            bottom[i].diff[...] = sign * (self.weight["w_a"] * self.grad_alpha + \
                                            self.weight["w_g"] * self.grad_grad)

    def alpha_prediction_loss(self, pred):
        # calculate alpha_prediction_loss here
        diff_ = pred - self.alpha
        # loss
        loss_ = np.sum((diff_**2) * self.mask) /\
                 (2 * (self.num_pixels + self.epsilon))
        self.diff_alpha = diff_

        # compute gradient
        self.grad_alpha = diff_ * self.mask / len(self.pred)
        return loss_

    def gradient_loss(self):
        diff_ = self.diff_alpha
        # compute loss
        loss_ = np.sum((diff_**2) * self.mask * self.gradient_map) /\
                (2 * (self.num_pixels + self.epsilon))
        
        # compute gradient
        self.grad_grad = diff_ * self.grad_alpha * self.gradient_map
        return loss_
