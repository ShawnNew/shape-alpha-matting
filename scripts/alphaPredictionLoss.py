import caffe
import numpy as np
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
        self.mask = bottom[1].data[:,0,:,:] / 255.
        self.alpha = bottom[1].data[:,1,:,:] / 255.
        self.gradient = bottom[1].data[:,2,:,:] / 255.
        self.pred = np.reshape(self.pred, (-1, 1, self.shape[0], self.shape[1]))
        self.mask = np.reshape(self.mask, (-1, 1, self.shape[0], self.shape[1]))
        self.alpha = np.reshape(self.alpha, (-1, 1, self.shape[0], self.shape[1]))
#        self.gradient = np.reshape(self.gradient, (-1, 1, self.shape[0], self.shape[1]))
        self.mask[self.mask == 0.] *= 0.
        self.mask[self.mask == 1.] *= 0.
        self.mask[self.mask != 0.] = 1.
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
            
            bottom[i].diff[...] = sign * self.diff / len(self.pred)

    def alpha_prediction_loss(self, pred):
        # calculate alpha_prediction_loss here
        self.diff = diff = (pred - self.alpha) * self.mask
        return np.sum(diff**2) / \
                (self.num_pixels + self.epsilon) / 2.

