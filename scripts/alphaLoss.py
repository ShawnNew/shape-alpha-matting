import caffe
import numpy as np

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
        self.shape = bottom[0][0][0].shape


    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # reshape the loss output
        top[0].reshape(1)

    def forward(self, bottom, top):
        # calculate loss here
        pred = bottom[0][:, 0, :, :]
        mask = bottom[1][:, 0, :, :]
        color_img = bottom[1][:, 1:4, :, :]
        alpha = bottom[1][:, 4, :, :]
        fg = bottom[1][:, 5:8, :, :]
        bg = bottom[1][:, 8:11, :, :]
        # self.diff[...] = pred - alpha
        top[0].data[...] = self.overall_loss(pred, mask, alpha, color_img, fg, bg)
        pred = np.reshape(pred, (-1, 1, self.shape[0], self.shape[1]))
        alpha = np.reshape(alpha, (-1, 1, self.shape[0], self.shape[1]))
        self.diff[...] = pred - alpha

    def backward(self, top, progagate_down, bottom):
        # calculate gradient here.
        # backpropogate the gradient
        for i in range(2):
            if not progagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff \
                                    / top[0].data \
                                    / bottom[i].num
    
    def alpha_prediction_loss(self, mask, pred, alpha):
        # calculate alpha_prediction_loss here
        diff = pred - alpha
        diff = diff * mask       # element-wise multiply
        num_pixels = np.sum(mask)
        return np.sum(np.sqrt(np.square(diff) + self.epsilon**2)) \
        / (num_pixels + self.epsilon) \
        / len(pred) # divid the batch size and the unknown region

    def compositional_loss(self, pred, mask, color_img, fg, bg):
        # calculate compositional_loss here
        mask = np.reshape(mask, (-1, 1, self.shape[0], self.shape[1]))
        pred = np.reshaple(pred, (-1, 1, self.shape[0], self.shape[1]))
        color_pred = pred * fg + (1.0 - pred) * bg    # element-wise multiply to get color image
        diff = color_pred - color_img
        diff = diff * mask
        num_pixels = np.sum(mask)
        return np.sum(np.sqrt(np.square(diff) + self.epsilon**2)) \
        / (num_pixels + self.epsilon) \ 
        / len(pred) # divide the batch size and the unknown region

    def overall_loss(self, pred, mask, alpha, color_img, fg, bg):
        # average the above two losses
        return self.w_l * self.alpha_prediction_loss(mask, pred, alpha) + \
                (1 - self.w_l) * self.compositional_loss(pred, mask, color_img, fg, bg)
        