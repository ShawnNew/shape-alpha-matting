import caffe
import cv2 as cv
import argparse
from utils import caffeTransform, greyscaleTransform
import pdb
import numpy as np
from PIL import Image
import os

parser = argparse.ArgumentParser()
# parser.add_argument("-m", "--model", \
#                     type=str,
#                     help="Please specify the model.")
# parser.add_argument("-w", "--weights", \
#                     type=str,
#                     help="Please specify the weights.")
parser.add_argument("-i", "--image", \
                    type=str,
                    help="Please specify the image used for inferencing.")
# parser.add_argument("-t", "--trimap", \
#                     type=str,
#                     help="Please specify the trimap used for inferencing.")
# parser.add_argument("-o", "--output", \
#                     type=str,
#                     help="Please specify the output of the image.")

args = parser.parse_args()
# model = args.model
image_path = args.image
# trimap_path = args.trimap
# weights = args.weights

model = "/home/dl/Codes/shape-alpha-matting/models/deploy.prototxt"
# pdb.set_trace()
base_image_path = "/home/dl/Codes/shape-alpha-matting/examples/original"
base_trimap_path = "/home/dl/Codes/shape-alpha-matting/examples/trimap"
weights = "/home/dl/Codes/shape-alpha-matting/models/snapshots/20190113/Training_iter_130000.caffemodel"
image_path = os.path.join(base_image_path, args.image)
trimap_path = os.path.join(base_trimap_path, args.image)

NN_SIZE = 224

caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net(model, weights, caffe.TEST)

image = cv.imread(image_path).astype(np.float64)
img_shape = [image.shape[0], image.shape[1]]
tri_map = cv.imread(trimap_path).astype(np.uint8)
tri_map = greyscaleTransform(tri_map)
image = caffeTransform(image, NN_SIZE)
tri_map = caffeTransform(tri_map, NN_SIZE)
image = np.reshape(image, (-1, 3, NN_SIZE, NN_SIZE))
tri_map = np.reshape(tri_map, (-1, 1, NN_SIZE, NN_SIZE))
net.blobs['data'].reshape(1, 3, NN_SIZE, NN_SIZE)
net.blobs['tri-map'].reshape(1, 1, NN_SIZE, NN_SIZE)
net.blobs['data'].data[...] = image
net.blobs['tri-map'].data[...] = tri_map
net.forward()
result = net.blobs['sigmoid_pred'].data * 255
result = np.reshape(result, (NN_SIZE, NN_SIZE)).astype(np.uint8)
result = cv.resize(result, (img_shape[1], img_shape[0]), interpolation=cv.INTER_CUBIC)
result_img_ = Image.fromarray(result)
result_path = os.path.join("../examples/result/", args.image)
result_img_.save(result_path)
