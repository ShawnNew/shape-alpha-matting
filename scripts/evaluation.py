import caffe
import time
import numpy as np

class ShapeAlphaNetModel:
    def __init__(self, model, weights, device, device_n):
        if device == "gpu":
            caffe.set_mode_gpu()
            caffe.set_device(device_n)
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(model, weights, caffe.TEST)

    def feed_input_with_shape(self, data):
        self.net.blobs['data'].data[...] = data[:, :3, :, :]
        self.net.blobs['tri-map'].data[...] = data[:, 3, :, :]
        self.net.blobs['gradient'].data[...] = data[:, 4, :, :]
#        self.net.blobs['roughness'].data[...] = data[:, 5, :, :]
        self.input_size_ = data[0][0].shape
    
    def predict_with_shape_data(self):
        t_start = time.clock()
        self.net.forward()
        duration = time.clock() - t_start
        raw_shape_output = self.net.blobs['sigmoid_pred'].data * 255.
        shape_output = self.net.blobs['alpha_output'].data * 255.
        shape_output = np.reshape(shape_output, \
                        (self.input_size_[0], self.input_size_[1])).astype(np.uint8)
        raw_shape_output = np.reshape(raw_shape_output, \
                           (self.input_size_[0], self.input_size_[1])).astype(np.uint8)
        return duration, shape_output, raw_shape_output


class alphaNetModel:
    def __init__(self, model, weights, device, device_n):
        if device == "gpu":
            caffe.set_mode_gpu()
            caffe.set_device(device_n)
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(model, weights, caffe.TEST)
    
    def feed_input(self, data):
        self.net.blobs['data'].data[...] = data[:, :3, :, :]
        self.net.blobs['tri-map'].data[...] = data[:, 3, :, :]
        self.input_size_ = data[0][0].shape

    def predict_without_shape_data(self):
        t_start = time.clock()
        self.net.forward()
        duration = time.clock() - t_start
        _output = self.net.blobs['alpha_output'].data * 255.
        _output = np.reshape(_output, \
                    (self.input_size_[0], self.input_size_[1])).astype(np.uint8)
        raw_output = self.net.blobs['sigmoid_pred'].data * 255.
        raw_output = np.reshape(raw_output, \
                               (self.input_size_[0], self.input_size_[1])).astype(np.uint8)
        return duration, _output, raw_output

