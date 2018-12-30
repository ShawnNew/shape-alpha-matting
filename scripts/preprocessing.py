import h5py
import os
import numpy as np
import caffe
import pdb
# from __future__ import print_function
DATASET = "/home/dl/Downloads/datasets/alphamatting/low-res/"
DIR = "../dataset"
RAW_IMAGE_SIZE = 256


def write_files(data, tri_map, label):
    h5_filename = os.path.join(DIR, 'alphamatting.h5')
    text_filename = os.path.join(DIR, 'alphamatting.txt')

    with h5py.File(h5_filename, 'w') as f:
        f['data'] = data
        f['tri-map'] = tri_map
        f['label'] = label

    with open(text_filename, 'w') as f:
        # print(h5_filename, file=f)
        f.write(h5_filename.replace('../dataset/', ''))


if __name__ == '__main__':
    fn = os.path.join(DATASET, 'train.txt')
    with open(fn, 'r') as f:
        data_lines = f.readlines()
        data = np.zeros(
            (len(data_lines), 3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE),\
            dtype='f4'
        )
        tri_map = np.zeros(
            (len(data_lines), 3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE), \
            dtype='f4'
        )
        label = np.zeros(
            (len(data_lines), 3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE), \
            dtype='f4'
        )
        for i, l in enumerate(data_lines):
            l = l.rstrip()
            l = l.replace('./', DATASET)
            items = l.split(' ')
            # pdb.set_trace()
            raw_img = caffe.io.load_image(items[0])
            data = caffe.io.resize(
                raw_img, (RAW_IMAGE_SIZE, RAW_IMAGE_SIZE, 3)
            )
            raw_tri_map = caffe.io.load_image(items[1])
            tri_map = caffe.io.resize(
                raw_tri_map, (RAW_IMAGE_SIZE, RAW_IMAGE_SIZE, 3)
            )
            raw_label = caffe.io.load_image(items[2])
            label = caffe.io.resize(
                raw_label, (RAW_IMAGE_SIZE, RAW_IMAGE_SIZE, 3)
            )
        write_files(data, tri_map, label)
