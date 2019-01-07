import os, glob
import pdb
import numpy as np
from utils import getFileList


DATASET_DIR = "/home/dl/Downloads/datasets/adobe"
shuffle = True
train_prop = 70
val_prop = 15
test_prop = 15

def writeDataSetFile(fn, perm):
    with open(fn, 'w') as f:
        for i in perm:
            line_ = trimap_list[i] + ' '  \
                    + original_list[i] + ' ' \
                    + gt_list[i] + ' ' \
                    + fg_list[i] + ' ' \
                    + bg_list[i] + '\n'
            f.write(line_)

if __name__ == '__main__':
    dir_list = os.listdir(DATASET_DIR)
    trimap_list = []
    original_list = []
    gt_list = []
    fg_list = []
    bg_list = []

    # get file list
    for sub in dir_list:
        if sub == 'TrimapImages':
            trimap_list = getFileList(DATASET_DIR, sub)
        elif sub == 'AlphaMatte':
            gt_list = getFileList(DATASET_DIR, sub)
        elif sub == 'Foreground':
            fg_list = getFileList(DATASET_DIR, sub)
        elif sub == 'OriginalImages':
            original_list = getFileList(DATASET_DIR, sub)
        elif sub == 'Background':
            bg_list = getFileList(DATASET_DIR, sub)
    
    # check
    if not (len(trimap_list) == len(original_list) \
            == len(gt_list) == len(fg_list) \
            == len(fg_list) == len(bg_list)):
        raise Exception("Number of images is not same.")

    if shuffle:
        perm = np.random.permutation(len(trimap_list))
        len_ = len(perm)
        train_perm_ = perm[0: len_ * train_prop / 100]
        val_perm_ = perm[len_ * train_prop / 100: \
                         len_ * (train_prop + val_prop) / 100]
        test_perm_ = perm[len_ * (train_prop + val_prop) / 100 :]
	train_path = os.path.join(DATASET_DIR, 'train.txt')
	val_path = os.path.join(DATASET_DIR, 'val.txt')
	test_path = os.path.join(DATASET_DIR, 'test.txt')
    writeDataSetFile(train_path, train_perm_)
    writeDataSetFile(val_path, val_perm_)
    writeDataSetFile(test_path, test_perm_)

    
