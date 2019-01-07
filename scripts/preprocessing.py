import h5py
import os, glob
import cv2
import numpy as np
import caffe
import argparse
import pdb
from utils import writeH5Files, writeH5TxtFile, caffeTransform, greyscaleTransform

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

if __name__ == '__main__':
    # arguments
    args = parser.parse_args()
    DATASET = args.directory
    OUTPUT_DIR = args.output
    RAW_IMAGE_SIZE = args.size
    SAMPLES = 256                 # samples each hdf5 file
    
    # main loop
    for file in os.listdir(DATASET):
        if file.endswith(".txt"):  # find all the txt files that contains the dataset
            # check and create output directory
            output_dir_ = os.path.join(OUTPUT_DIR, file.replace(".txt", ""))
            dataset_file_ = os.path.join(DATASET, file)
            if not os.path.exists(output_dir_):
                os.makedirs(output_dir_)
            
            print "Dataset file name is:", dataset_file_
            print "Writing hdf5 files into:", output_dir_
            with open(dataset_file_, 'r') as f:
                lines_ = f.readlines()
                file_count = 0
                data = np.zeros(      # original data (color)
                    (SAMPLES, 3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE),\
                    dtype='f4')
                tri_map = np.zeros(   # tri-map is greyscale images
                    (SAMPLES, 1, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE), \
                    dtype='f4')
                gt = np.zeros(        # ground truth is greyscale images
                    (SAMPLES, 1, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE), \
                    dtype='f4')
                fg = np.zeros(        # foreground (color)
                    (SAMPLES, 3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE), \
                    dtype='f4')
                bg = np.zeros(        # foreground (color)
                    (SAMPLES, 3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE), \
                    dtype='f4')
                print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

                for i, l in enumerate(lines_):
                    # parse data directory
                    items = l.rstrip().replace('./', DATASET).split(' ')
                    img_count_ = i%SAMPLES
                    # load raw dataset images
                    #TODO: the channel order may not align with vgg-16 pre-trained model
                    # here in opencv, the channel order is BGR
                    if not len(items) == 5:
                        raise Exception("columns of each line is not right.")
                    
                    raw_tri_map_ = cv2.imread(items[0]).astype(np.uint8)
                    raw_img_ = cv2.imread(items[1]).astype(np.float64)
                    raw_gt_ = cv2.imread(items[2]).astype(np.uint8)
                    raw_fg_ = cv2.imread(items[3]).astype(np.float64)
                    raw_bg_ = cv2.imread(items[4]).astype(np.float64)

                    # transform tri-map and gt to greyscale
                    raw_tri_map_ = greyscaleTransform(raw_tri_map_)
                    raw_gt_ = greyscaleTransform(raw_gt_)

                    # caffe transform
                    tri_map[img_count_] = caffeTransform(raw_tri_map_, RAW_IMAGE_SIZE)
                    data[img_count_] = caffeTransform(raw_img_, RAW_IMAGE_SIZE)
                    gt[img_count_] = caffeTransform(raw_gt_, RAW_IMAGE_SIZE)
                    fg[img_count_] = caffeTransform(raw_fg_, RAW_IMAGE_SIZE)
                    bg[img_count_] = caffeTransform(raw_bg_, RAW_IMAGE_SIZE)
                    print "Processed %d images"% int(i+1)
                    
                    if img_count_+1 == SAMPLES: # write SAMPLES images into hdf5 file
                        file_count += 1
                        filepath = os.path.join(output_dir_, file.replace(".txt", "")+str(file_count))
                        print "Writing file:", filepath
                        writeH5Files(OUTPUT_DIR, data, tri_map, gt, fg, bg, filepath)
                filepath = os.path.join(output_dir_, file.replace(".txt", "")+str(file_count+1))
                print "Writing file:", filepath
                writeH5Files(OUTPUT_DIR, data, tri_map, gt, fg, bg, filepath)
                print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

            # writeH5Files(OUTPUT_DIR, data, tri_map, gt, fg, bg, file.replace(".txt", ""))
