import h5py
import os
import cv2
import numpy as np
import caffe
import pdb
DATASET = "/home/dl/Downloads/datasets/alphamatting/low-res/"
DIR = "../dataset"
RAW_IMAGE_SIZE = 224
h5_filename = 'alphamatting.h5'
text_filename = 'alphamatting.txt'
shuffle = True

def write_files(data, tri_map, label):    
    text_filedir = os.path.join(DIR, text_filename)
    h5_filedir = os.path.join(DIR, h5_filename)
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
    fn = os.path.join(DATASET, 'train.txt')
    print "Dataset directiry is:", DATASET
    img_count = 1
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
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        for i, l in enumerate(data_lines):
            print "Preprocessed %d images"% img_count
            # parse data directory
            items = l.rstrip().replace('./', DATASET).split(' ')
            # load raw dataset images
            #TODO: the channel order may not align with vgg-16 pre-trained model
            # here in opencv, the channel order is BGR
            raw_img = cv2.imread(items[0]).astype(np.float64)
            raw_tri_map = cv2.imread(items[1]).astype(np.float64)
            raw_label = cv2.imread(items[2]).astype(np.float64)
            # pre-process
            data[i] = caffe_transform(raw_img)
            tri_map[i] = caffe_transform(raw_tri_map)
            label[i] = caffe_transform(raw_label)
            img_count += 1
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "All %d images are been processed, write to hdf5 file(%s).\n"% \
            (img_count-1, h5_filename)
        # pdb.set_trace()

        # shuffle
        if shuffle:
            perm = np.random.permutation(len(data_lines))
            data = data[perm]
            tri_map = tri_map[perm]
            label = label[perm]
        write_files(data, tri_map, label)
