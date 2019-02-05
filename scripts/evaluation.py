import caffe
import cv2 as cv
from PIL import Image
import configparser
from scipy import misc as misc
import pdb
import numpy as np
import os


shape_model="/home/dl/Codes/shape-alpha-matting/models/shapeAlphaMattingNet_deploy.prototxt"
_model="/home/dl/Codes/shape-alpha-matting/models/alphaMattingNet_deploy.prototxt"

shape_weights="/home/dl/Codes/shape-alpha-matting/models/snapshots/shapeAlphaNet/all/Refine_training_iter_250000.caffemodel"
_weights="/home/dl/Codes/shape-alpha-matting/models/snapshots/alphaNet/all/Refine_training_iter_250000.caffemodel"

source="/home/dl/harddisk/Datasets/adobe-h5/test.txt"

class ShapeAlphaNetModel:
    def __init__(self, model, weights, device, device_n):
        if device == "gpu":
            caffe.set_gpu()
            caffe.set_device(device_n)
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(model, weights, caffe.TEST)

    def feed_input_with_shape(self, data):
        self.net.blobs['data'] = data[:3, :, :]
        self.net.blobs['tri-map'] = data[3, :, :]
        self.net.blobs['gradient'] = data[4, :, :]
        self.net.blobs['roughness'] = data[5, :, :]

class alphaNetModel:
    def __init__(self, model, weights, device, device_n):
        if device == "gpu":
            caffe.set_mode_gpu()
            caffe.set_device(device_n)
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(model, weights, caffe.TEST)
    
    def feed_input(self, data):
        self.net.blobs['data'] = data[:3, :, :]
        self.net.blobs['tri-map'] = data[3, , :]

if __name__ == "__main__":
    shape_model = ShapeAlphaNetModel(shape_model, shape_weights, 'gpu', 2)
    _model = alphaNetModel(_model, _weights, 'gpu', 3)
    shape_mse = _mse = 0
    with open(source, 'r') as f:
        lines_ = f.readlines():
        nums = len(lines_)
        for i, l in enumerate(lines_):
            items = l.rstrip().replace('./', os.path.split(source)).split(' ')
            # read images from datasets directory
            data = misc.imresize(
                    np.asarray(cv2.imread(items[6])), \
                    [224, 224], \
                    interp='nearest').astype(np.float64)
            tri_map = misc.imresize(
                    np.asarray(Image.open(items[1])), \
                    [224, 224], \
                    interp='nearest').astype(np.float64)
            gradient = misc.imresize(
                    np.asarray(Image.open(items[2])), \
                    [224, 224], \
                    interp='nearest').astype(np.float64)
            roughness = misc.imresize(np.asarray(
                    Image.open(items[3])), \
                    [224, 224], \
                    interp='nearest').astype(np.float64)
            gt = np.asarray(Image.open(items[4])).astype(np.float64)
            tri_map = np.expand_dims(tri_map, axis=2)
            gradient = np.expand_dims(gradient, axis=2)
            roughness = np.expand_dims(roughness, axis=2)

            shape_ = gt.shape
            # test shape data
            feed_data_with_shape = np.concatenate([data, tri_map, gradient, roughness,
                                        axis=0)
            feed_data_with_shape = np.transpose(feed_data_with_shape, (2, 0, 1))
            shape_model.feed_input_with_shape(feed_data)
            shape_model.net.forward()
            shape_output = shape_model.net.blobs['alpha_output'].data * 255.
            shape_output = misc.resize(shape_output, [shape_[0], shape_[1]], interp='nearest')
            shape_mse += np.sum((shape_output - gt) ** 2)
            shape_output_img = Image.fromarray(shape_output)
            shape_test_output_dir = os.path.join(os.path.split(source), 'shape-test-output')
            shape_test_output_filename = os.path.join(shape_test_output_dir, os.path.basename(items[0]))
            shape_output_img.save(shape_test_output_filename)
            


            # test without shape data
            feed_data_without_shape = np.concatenate([data, tri_map, axis=0)
            feed_data_without_shape = np.transpose(feed_data_without_shape, (2, 0, 1))
            _model.feed_input(feed_data_without_shape)
            _output = _model.net.blobs['alpha_output'].data * 255.
            _output = misc.resize(_output, [shape_[0], shape_[1]], interp='nearest')
            _mse += np.sum((_output - gt) ** 2)
            _output_img = Image.fromarray(_output)
            _output_test_output_dir = os.path.join(os.path.split(source), 'test-output')
            _output_test_output_filename = os.path.join(_output_test_output_dir, os.path.basename(items[0]))
            _output_img.save(_output_test_output_filename)
        
        shape_mse = shape_mse / (2 * nums)
        _mse = _mse / (2 * nums)
        print "shape mse is:", shape_mse
        print "mse is:", _mse