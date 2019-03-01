import caffe
import numpy as np
import utils
from utils import generate_gradient_map
import pdb

class EncoderDecoderLossLayer(caffe.Layer):
    """
    Loss used in the first state of alpha matting.
    It is a combine of alpha-loss and compositional-loss.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        params = eval(self.param_str)
        w_ = {}
        w_["w_a"] = params["w_a"]
        w_["w_c"] = params["w_c"]
        self.weights = w_
        self.epsilon = params["epsilon"]
        self.shape = bottom[0].data[0][0].shape

    def reshape(self, bottom, top):
        # reshape the inter-mediate data from the blobs here.
        # loss output is scalar
        top[0].reshape(1)
        # reshape the data
        self.pred = bottom[0].data[:,0,:,:]
        mask_ = bottom[1].data[:,0,:,:] / 255.
        self.color_img = bottom[1].data[:,1:4,:,:] / 255.
        self.alpha = bottom[1].data[:,4,:,:] / 255.
        self.fg = bottom[1].data[:,5:8,:,:] / 255.
        self.bg = bottom[1].data[:,8:11,:,:] / 255.
        self.pred = np.reshape(self.pred, (-1, 1, self.shape[0], self.shape[1]))
        mask_ = np.reshape(mask_, (-1, 1, self.shape[0], self.shape[1]))
        self.alpha = np.reshape(self.alpha, (-1, 1, self.shape[0], self.shape[1]))
        mask_[mask_ == 0.] *= 0.
        mask_[mask_ == 1.] *= 0.
        mask_[mask_ != 0.] = 1.
        self.mask = mask_
        self.num_pixels = np.sum(self.mask)

    def forward(self, bottom, top):
        top[0].data[...] = self.overall_loss(self.pred)
    
    def backward(self, top, propagate_down, bottom):
        _diff = self.weights["w_a"] * self.grad_alpha +\
                self.weights["w_c"] * self.grad_comp
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            
            bottom[0].diff[...] = sign * _diff
    
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

    
    
    def compositional_loss(self, pred):
        # calculate compositional_loss here
        color_pred = pred * self.fg + (1.0 - pred) * self.bg      
        diff_ = color_pred - self.color_img
        diff_average = np.average(diff_, axis=1)   # average over color channel
        diff_average = np.reshape(diff_average, (-1, 1, self.shape[0], self.shape[1]))
        # compute loss here
        loss_ = np.sum((diff_average**2) * self.mask) /\
                (2 * (self.num_pixels + self.epsilon))
        self.diff_comp = diff_average
        
        # compute gradient here
        grad_comp_ = np.average(diff_ * (self.fg-self.bg), axis=1)
        grad_comp_ = np.reshape(grad_comp_, (-1, 1, self.shape[0], self.shape[1]))
        self.grad_comp = grad_comp_ * self.mask /\
                         len(self.pred)
        return loss_


    def overall_loss(self, pred):                
        # average the above two losses
        alpha_loss_ = self.alpha_prediction_loss(pred)
        comp_loss_ = self.compositional_loss(pred)
        return self.weights["w_a"] * alpha_loss_ + \
                self.weights["w_c"] * comp_loss_

class EncoderDecoderWithGradientLossLayer(caffe.Layer):
    """
    Loss used in the first state of alpha matting.
    It is a combine of alpha-loss and compositional-loss.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        params = eval(self.param_str)
        w_ = {}
        w_["w_a"] = params["w_a"]
        w_["w_c"] = params["w_c"]
        w_["w_g"] = params["w_g"]
        self.weights = w_
        self.epsilon = params["epsilon"]
        self.shape = bottom[0].data[0][0].shape

    def reshape(self, bottom, top):
        # reshape the inter-mediate data from the blobs here.
        # loss output is scalar
        top[0].reshape(1)
        # reshape the data
        self.pred = bottom[0].data[:,0,:,:]
        mask_ = bottom[1].data[:,0,:,:] / 255.
        self.color_img = bottom[1].data[:,1:4,:,:] / 255.
        self.alpha = bottom[1].data[:,4,:,:] / 255.
        self.fg = bottom[1].data[:,5:8,:,:] / 255.
        self.bg = bottom[1].data[:,8:11,:,:] / 255.
        gradient = bottom[1].data[:,11,:,:] / 255.
        self.pred = np.reshape(self.pred, (-1, 1, self.shape[0], self.shape[1]))
        mask_ = np.reshape(mask_, (-1, 1, self.shape[0], self.shape[1]))
        self.alpha = np.reshape(self.alpha, (-1, 1, self.shape[0], self.shape[1]))
        gradient = np.reshape(gradient, (-1, 1, self.shape[0], self.shape[1]))
        self.gradient_map = generate_gradient_map(gradient, 3)
        mask_[mask_ == 0.] *= 0.
        mask_[mask_ == 1.] *= 0.
        mask_[mask_ != 0.] = 1.
        self.mask = mask_
        self.num_pixels = np.sum(self.mask)

    def forward(self, bottom, top):
        top[0].data[...] = self.overall_loss(self.pred)
    
    def backward(self, top, propagate_down, bottom):
        _diff = self.weights["w_a"] * self.grad_alpha +\
                self.weights["w_c"] * self.grad_comp +\
                self.weights["w_g"] * self.grad_grad
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            
            bottom[0].diff[...] = sign * _diff
    
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

    
    
    def compositional_loss(self, pred):
        # calculate compositional_loss here
        color_pred = pred * self.fg + (1.0 - pred) * self.bg      
        diff_ = color_pred - self.color_img
        diff_average = np.average(diff_, axis=1)   # average over color channel
        diff_average = np.reshape(diff_average, (-1, 1, self.shape[0], self.shape[1]))
        # compute loss here
        loss_ = np.sum((diff_average**2) * self.mask) /\
                (2 * (self.num_pixels + self.epsilon))
        self.diff_comp = diff_average
        
        # compute gradient here
        grad_comp_ = np.average(diff_ * (self.fg-self.bg), axis=1)
        grad_comp_ = np.reshape(grad_comp_, (-1, 1, self.shape[0], self.shape[1]))
        self.grad_comp = grad_comp_ * self.mask /\
                         len(self.pred)
        return loss_


    def gradient_loss(self):
        diff_ = 0.5 * self.diff_alpha + 0.5 * self.diff_comp
        # compute loss
        loss_ = np.sum((diff_**2) * self.mask * self.gradient_map) /\
                (2 * (self.num_pixels + self.epsilon))
        
        # compute gradient
        self.grad_grad = diff_ * (0.5 * self.grad_alpha + 0.5 * self.grad_comp) *\
                self.gradient_map
        return loss_

    def overall_loss(self, pred):                
        # average the above two losses
        alpha_loss_ = self.alpha_prediction_loss(pred)
        comp_loss_ = self.compositional_loss(pred)
        gradient_loss_ = self.gradient_loss()
        return self.weights["w_a"] * alpha_loss_ + \
                self.weights["w_c"] * comp_loss_ +\
                self.weights["w_g"] * gradient_loss_
