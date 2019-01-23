import os
import h5py
import cv2
import numpy as np
from utils import *


class Preprocessor:
    # class member
    samples_ = 256
    ele_ = 5    # trimap, foreground, background, gt, original
    random_list_ = [320, 480, 640]
    flip_list_ = [True, False]
    # class constructor
    def __init__(self, root, output, size):
        self.root_dir_ = root
        self.output_dir_ = output
        self.img_size_ = size
    
    def getSplitedDatasetList(self):
        """Parse the text file in the root of the dataset,
        which contains train, test, validation split .txt files.
        
        In each txt file, there are samples organized as rows.
        """
        list_ = []
        for file in os.listdir(self.root_dir_):
            if file.endswith(".txt"):
                split_dir_ = os.path.join(self.root_dir_, file)
                list_.append(split_dir_)
        self.dataset_split_list_ = list_

    
    def splitDataset(self, \
                    prop_train=0.7, \
                    prop_test=0.15, \
                    prop_val=0.15, \
                    shuffle=True):
        def writeDataSetFile(fn, perm):
            with open(fn, 'w') as f:
                for i in perm:
                    line_ = trimap_list[i] + ' '  \
                            + original_list[i] + ' ' \
                            + gt_list[i] + ' ' \
                            + fg_list[i] + ' ' \
                            + bg_list[i] + '\n'
                    f.write(line_)
        for item in os.listdir(self.root_dir_):
            _path = os.path.join(self.root_dir_, item)
            if (os.path.isdir(_path)):
                if item == 'TrimapImages':
                    trimap_list = getFileList(self.root_dir_, item)
                elif item == 'AlphaMatte':
                    gt_list = getFileList(self.root_dir_, item)
                elif item == 'Foreground':
                    fg_list = getFileList(self.root_dir_, item)
                elif item == 'OriginalImages':
                    original_list = getFileList(self.root_dir_, item)
                elif item == 'Background':
                    bg_list = getFileList(self.root_dir_, item)
        # check
        if not (len(trimap_list) == len(original_list) \
                == len(gt_list) == len(fg_list) \
                == len(fg_list) == len(bg_list)):
            raise Exception("Number of images is not same.")
        if shuffle:
            perm = np.random.permutation(len(trimap_list))
            len_ = len(perm)
            train_perm_ = perm[0: len_ * prop_train]
            val_perm_ = perm[len_ * prop_train: \
                            len_ * (prop_train + prop_val)]
            test_perm_ = perm[len_ * (prop_train + prop_val):]
        train_path = os.path.join(self.root_dir_, 'train.txt')
        val_path = os.path.join(self.root_dir_, 'val.txt')
        test_path = os.path.join(self.root_dir_, 'test.txt')
        writeDataSetFile(train_path, train_perm_)
        writeDataSetFile(val_path, val_perm_)
        writeDataSetFile(test_path, test_perm_)
        self.getSplitedDatasetList() # return dataset file lists.

    
    def writeHDF5Files(self):
        for file in self.dataset_split_list_:
            dataset_file_ = os.path.join(self.root_dir_, file)
            output_file_path_ = os.path.join(self.output_dir_, file.replace(".txt", ""))
            print "Dataset file name is:", dataset_file_
            with open(dataset_file_, 'r') as f:
                lines_ = f.readlines()
                file_count = 0
                data = np.zeros(      # original data (color)
                    (self.samples_, 3, self.img_size_, self.img_size_),\
                    dtype='f4')
                tri_map = np.zeros(   # tri-map is greyscale images
                    (self.samples_, 1, self.img_size_, self.img_size_), \
                    dtype='f4')
                gt = np.zeros(        # ground truth is greyscale images
                    (self.samples_, 1, self.img_size_, self.img_size_), \
                    dtype='f4')
                fg = np.zeros(        # foreground (color)
                    (self.samples_, 3, self.img_size_, self.img_size_), \
                    dtype='f4')
                bg = np.zeros(        # foreground (color)
                    (self.samples_, 3, self.img_size_, self.img_size_), \
                    dtype='f4')
                print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

                for i, l in enumerate(lines_):
                    # parse data directory
                    items = l.rstrip().replace('./', self.root_dir_).split(' ')
                    img_count_ = i%(self.samples_)
                    # load raw dataset images
                    #TODO: the channel order may not align with vgg-16 pre-trained model
                    # here in opencv, the channel order is BGR
                    if not len(items) == self.ele_:
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
                    tri_map[img_count_] = caffeTransform(raw_tri_map_, self.img_size_)
                    data[img_count_] = caffeTransform(raw_img_, self.img_size_)
                    gt[img_count_] = caffeTransform(raw_gt_, self.img_size_)
                    fg[img_count_] = caffeTransform(raw_fg_, self.img_size_)
                    bg[img_count_] = caffeTransform(raw_bg_, self.img_size_)
                    print "Processed %d samples"% int(i+1)
                    
                    if img_count_+1 == self.samples_: # write SAMPLES images into hdf5 file
                        file_count += 1
                        filename = file.replace(".txt", "") + str(file_count) + ".h5"
                        filepath = os.path.join(output_file_path_, filename)
                        print "Writing file:", filepath
                        writeH5Files(self.output_dir_, data, tri_map, gt, fg, bg, filepath)
                filename = file.replace(".txt", "") + str(file_count+1) +  ".h5"
                filepath = os.path.join(output_file_path_, filename)
                print "Writing file:", filepath
                writeH5Files(self.output_dir_, data, tri_map, gt, fg, bg, filepath)
                print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            # write hdf5 files list into txt file, which will be used in caffe data layer
            writeH5TxtFile(self.output_dir_, file.replace(".txt", ""))
        

    def imgCropper(imgs, trimap, crop_size, flip_flag, output_size):
        #TODO: write crop function.
