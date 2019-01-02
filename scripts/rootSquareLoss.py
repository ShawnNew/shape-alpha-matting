import caffe
import numpy as np

class RootSquareLossLayer(caffe.Layer):
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
            raise Exception("Wrong number of bottom blobs (prediction and label)")
        if bottom[0].data.ndim != bottom[1].data.ndim:
            raise Exception("Prediction's dimension does not align with the label")


    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # reshape the loss output
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(np.sqrt(np.square(self.diff) + 1e-6))

    def backward(self, top, progagate_down, bottom):
        for i in range(2):
            if not progagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num