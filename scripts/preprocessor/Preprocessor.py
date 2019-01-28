import os
import h5py
import cv2
import numpy as np
from PIL import Image
import sys
from utils import *


class Preprocessor:
    # class member
    channels = 11 # the total channel number of the dataset
    random_crop_list_ = [320, 480, 640]
    flip_list_ = [True, False]
    
    
    # class constructor
    def __init__(self, root, output, size, samples_per_file):
        self.root_dir_ = root
        self.output_dir_ = output
        self.img_size_ = size
        self.samples_ = samples_per_file

    def parseDatasetDirectory(self):
        """
        Parse the dataset directory and record the relative path of images 
        within each folder into a dictionary.
        """
        _dict = {}
        for item in os.listdir(self.root_dir_):
            _path = os.path.join(self.root_dir_, item)
            if (os.path.isdir(_path)):  
                _dict[item] = getFileList(self.root_dir_, item)  # relative path against root
        
        self.dataset_dict = _dict
        self.ele_ = len(_dict.keys())


    def getSplitedDataset(self, \
                    prop_train=0.7, \
                    prop_test=0.15, \
                    prop_val=0.15, \
                    shuffle=True):
        """
        Shuffle, split the dataset and record the index as txt files.
        """
        def writeDataSetFile(dict_, fns, list_):
            for fn in fns:
                with open(fn, 'w') as f:
                    for item in list_:
                        # f.write(dict_.values)
                        for key in dict_.keys():
                            path, filename = os.path.split(dict_[key][0])
                            path = os.path.join(path, item)
                            f.write(path + ' ')
                        f.write('\n')

        # count sample number according to the smallest folder
        small_count = sys.maxint
        small_folder = " "
        for key, value in self.dataset_dict.iteritems():
            len_ = len(value)
            if (small_count > len_):
                small_count = min(small_count, len_)
                small_folder = key
            else:
                pass
        
        files_list = self.dataset_dict[small_folder]  # file relative path list
        if shuffle:
            random.shuffle(files_list)

        train_path = os.path.join(self.root_dir_, 'train.txt')
        val_path = os.path.join(self.root_dir_, 'val.txt')
        test_path = os.path.join(self.root_dir_, 'test.txt')
        list_ = [train_path, val_path, test_path]
        writeDataSetFile(self.dataset_dict, \
                        list_, \
                        files_list)
        self.dataset_split_list_ = list_

    
    def writeHDF5Files(self, scale=1):
        for file in self.dataset_split_list_:
            dataset_file_ = os.path.join(self.root_dir_, file)
            output_file_path_ = os.path.join(self.output_dir_, file.replace(".txt", ""))
            print "Dataset file name is:", dataset_file_
            with open(dataset_file_, 'r') as f:
                lines_ = f.readlines()
                file_count = 0
                data = np.zeros(
                    (self.samples_, self.channels, self.img_size_, self.img_size_), \
                    dtype='f4'
                )
                print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

                for i, l in enumerate(lines_):
                    # parse data directory
                    items = l.rstrip().replace('./', self.root_dir_).split(' ')
                    sample_count_ = i%(self.samples_)
                    if not len(items) == self.ele_:
                        raise Exception("columns of each line is not right.")
                    raw_tri_map_ = np.asarray(Image.open(items[0])) # 2-D as grey image
                    raw_img_ = np.asarray(cv2.imread(items[1]))
                    raw_gt_ = np.asarray(Image.open(items[2]))      # 2-D as grey
                    raw_fg_ = np.asarray(cv2.imread(items[3]))
                    raw_bg_ = np.asarray(cv2.imread(items[4]))

                    sample_array = np.concatenate([raw_img_, \
                                                np.expand_dims(raw_tri_map_, axis=2), \
                                                np.expand_dims(raw_gt_, axis=2), \
                                                raw_fg_, \
                                                raw_bg_], axis=2).astype(np.float64)

                    sample_array = self.imgCropper(sample_array, \
                                                np.expand_dims(raw_tri_map_, axis=2), \
                                                self.img_size_)
                    flip = random.choice(self.flip_list_)
                    if flip:
                        sample_array = sample_array[:, ::-1, :]
                    data[sample_count_] = np.transpose(sample_array, (2, 0, 1)) * scale


                    print "Processed %d samples"% int(i+1)
                    if sample_count_+1 == self.samples_: # write SAMPLES images into hdf5 file
                        file_count += 1
                        filename = file.replace(".txt", "") + str(file_count) + ".h5"
                        filepath = os.path.join(output_file_path_, filename)
                        print "Writing file:", filepath
                        writeH5Files(self.output_dir_, data, filepath)
                filename = file.replace(".txt", "") + str(file_count+1) +  ".h5"
                filepath = os.path.join(output_file_path_, filename)
                print "Writing file:", filepath
                writeH5Files(self.output_dir_, data, filepath)
                print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            # write hdf5 files list into txt file, which will be used in caffe data layer
            writeH5TxtFile(self.output_dir_, file.replace(".txt", ""))
        

    def imgCropper(self, img, trimap, output_size):
        random_cropsize = random.choice(self.random_crop_list_)
        original_shape = trimap.shape
        # resize if the original image is not large enough
        if(min(original_shape[0], original_shape[1])<random_cropsize):
            if(original_shape[0] < original_shape[1]):
                ratio = random_cropsize / original_shape[0]
                h = int(original_shape[0]*ratio+1)
                w = int(original_shape[1]*ratio)
            else:
                ratio = random_cropsize / original_shape[1]
                h = int(original_shape[0]*ratio)
                w = int(original_shape[1]*ratio+1)
            img = batch_resize(img, h, w)

        half, valid_unknown_area = validUnknownRegion(trimap, random_cropsize)
        if random_cropsize != output_size:
            img = img[
                valid_unknown_area[0]-half : valid_unknown_area[0]+half, \
                valid_unknown_area[1]-half : valid_unknown_area[1]+half, :]
            img = batch_resize(img, output_size, output_size)
        else:
            img = img[
                valid_unknown_area[0]-half : valid_unknown_area[0]+half, \
                valid_unknown_area[1]-half : valid_unknown_area[1]+half, :]
        return img