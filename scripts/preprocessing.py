import h5py
import os, glob
import cv2
import numpy as np
import caffe
import argparse
import pdb
"""
This script is used to parse the alpha-matting dataset.
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", \
                    type=str,
                    help="Please specify the directory of the images dataset.")
parser.add_argument("-o", "--output", \
                    type=str, \
                    default="../dataset", \
                    help="Please specify the directory of the output.")
parser.add_argument("-s", "--size", \
                    type=int, \
                    help="Please specify the output image size.")
# parser.add_argument("-n", "--name", \
#                     type=str, \
#                     help="Please specify the output filename.")
parser.add_argument("-r", "--random", \
                    type=bool, \
                    default=True, \
                    help="Please indicate wether shuffle or not.")

args = parser.parse_args()
DATASET = args.directory
OUTPUT_DIR = args.output
RAW_IMAGE_SIZE = args.size
shuffle = args.random

def write_files(data, tri_map, label, fn):
    h5_filename = fn + ".h5"
    text_filename = fn + ".txt"    
    text_filedir = os.path.join(OUTPUT_DIR, text_filename)
    h5_filedir = os.path.join(OUTPUT_DIR, h5_filename)
    hdf_file = h5py.File(h5_filedir, 'w')
    datum_shape = data[0].shape
    chunks = (1, datum_shape[0], datum_shape[1], datum_shape[2])
    hdf_file.create_dataset('data', dtype=np.float, data=data, chunks=chunks)
    hdf_file.create_dataset('tri-map', dtype=np.float, data=tri_map, chunks=chunks)
    hdf_file.create_dataset('label', dtype=np.float, data=label, chunks=chunks)
    hdf_file.flush()
    hdf_file.close()

    with open(text_filedir, 'w') as f:
        f.write(h5_filedir)

def caffe_transform(img):
    img = cv2.resize(img, \
                    (RAW_IMAGE_SIZE, RAW_IMAGE_SIZE),\
                    interpolation=cv2.INTER_CUBIC
                )
    img = np.transpose(img, (2, 0, 1))
    #TODO: normalization to be done here
    return img / 255.

if __name__ == '__main__':
    
    for file in os.listdir(DATASET):
        if file.endswith(".txt"):
            fn = os.path.join(DATASET, file)
            print "Dataset directiry is:", fn
            img_count = 1
            with open(fn, 'r') as f:
                lines_ = f.readlines()
                data = np.zeros(
                    (len(lines_), 3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE),\
                    dtype='f4')
                tri_map = np.zeros(
                    (len(lines_), 3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE), \
                    dtype='f4')
                label = np.zeros(
                    (len(lines_), 3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE), \
                    dtype='f4')
                print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                for i, l in enumerate(lines_):
                    # parse data directory
                    items = l.rstrip().replace('./', DATASET).split(' ')
                    # pdb.set_trace()
                    # load raw dataset images
                    #TODO: the channel order may not align with vgg-16 pre-trained model
                    # here in opencv, the channel order is BGR
                    if len(items) == 3:
                        raw_img = cv2.imread(items[0]).astype(np.float64)
                        raw_tri_map = cv2.imread(items[1]).astype(np.float64)
                        raw_label = cv2.imread(items[2]).astype(np.float64)
                        # pre-process
                        data[i] = caffe_transform(raw_img)
                        tri_map[i] = caffe_transform(raw_tri_map)
                        label[i] = caffe_transform(raw_label)
                    elif len(items) == 2:
                        raw_img = cv2.imread(items[0]).astype(np.float64)
                        raw_tri_map = cv2.imread(items[1]).astype(np.float64)
                        data[i] = caffe_transform(raw_img)
                        tri_map[i] = caffe_transform(raw_tri_map)
                    print "Preprocessed %d images"% img_count
                    img_count += 1
                print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

                # shuffle
                if shuffle:
                    perm = np.random.permutation(len(lines_))
                    data = data[perm]
                    tri_map = tri_map[perm]
                    label = label[perm] # label will be zero for test set.
            write_files(data, tri_map, label, file.replace(".txt", ""))
