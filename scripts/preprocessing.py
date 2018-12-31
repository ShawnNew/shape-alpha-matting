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
        f.create_dataset('data', data=data)
        f.create_dataset('tri-map', data=tri_map)
        f.create_dataset('label', data=label)

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
            raw_img = caffe.io.load_image(items[0])
            #TODO: img is not resize correctly here
            raw_img = caffe.io.resize(
                raw_img, (3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE)
            )
            
            raw_tri_map = caffe.io.load_image(items[1])
            raw_tri_map = caffe.io.resize(
                raw_tri_map, (3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE)
            )
            raw_label = caffe.io.load_image(items[2])
            raw_label = caffe.io.resize(
                raw_label, (3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE)
            )
            data[i] = raw_img
            tri_map[i] = raw_tri_map
            label = raw_label
            pdb.set_trace()
        # pdb.set_trace()
        write_files(data, tri_map, label)
