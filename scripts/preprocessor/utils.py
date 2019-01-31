import os
import h5py
import cv2
import numpy as np
import random
from scipy import misc as misc
import math
import pdb

def writeH5Files(out_dir, samples_array, file):
    """
    Write the formatted data into hdf5 file from img_array
    """
    hdf_file = h5py.File(file, 'w')
    shape = samples_array[0].shape
    chunks_data = (1, shape[0], shape[1], shape[2])
    hdf_file.create_dataset('data', dtype=np.float, data=samples_array, chunks=chunks_data)
    hdf_file.flush()
    hdf_file.close()


def writeH5TxtFile(out_dir, h5dir):
    """
    Generate a txt file that list all the HDF5 files of the dataset.
    Param:
        out_dir: the output directory
        h5dir:   the hdf5 dataset directory
    """
    text_filename = h5dir + ".txt"
    text_filename = os.path.join(out_dir, text_filename)
    text_filespath = os.path.join(out_dir, h5dir)
    files = os.listdir(text_filespath)
    with open(text_filename, 'w') as f:
        for file in files:
            line_ = text_filespath + '/' + file + '\n'
            f.write(line_)

def getFileList(base, sub):
    """
    Get file list of a directory:
    Param:
        base: base directory
        sub: sub-directory name
    Return:
        a list of image file name
    """
    path = os.path.join(base, sub)
    files = os.listdir(path)
    fileList = []
    for f in files:
        if (os.path.isfile(path + '/' + f)):
            path_ = './' + sub
            path_ = os.path.join(path_, f)
            # add image file into list
            fileList.append(path_)
    return fileList  


def candidateUnknownRegion(img):
    '''
    Propose a condidate of unknown region center randomly within the unknown area of img.
    param: 
        img: trimap image
    return: 
        an index for unknown region
    '''
    index = np.where(img == 128)
    i = random.choice([j for j in range(len(index[0]))])
    return np.array(index)[:, i][:2]

def validUnknownRegion(img, output_size):
    """
    Check wether the candidate unknown region is valid and return the index.
    param:
        img:            trimap image
        output_size:    the desired output image size
    return:
        output the crop start and end index along h and w respectively.
    """
    h_start = h_end = w_start = w_end = 0
    cand = candidateUnknownRegion(img)
    shape_ = img.shape
    if (output_size == 320):
        h_start = max(cand[0]-159, 0)
        w_start = max(cand[1]-159, 0)
        if (h_start+320 > shape_[0]):
            h_start = shape_[0] - 320
        if (w_start+320 > shape_[1]):
            w_start = shape_[1] - 320
        h_end = h_start + 320
        w_end = w_start + 320
        return h_start, h_end, w_start, w_end
    elif (output_size == 480):
        h_start = max(cand[0]-239, 0)
        w_start = max(cand[1]-239, 0)
        if (h_start+480 > shape_[0]):
            h_start = shape_[0] - 480
        if (w_start+480 > shape_[1]):
            w_start = shape_[1] - 480
        h_end = h_start + 480
        w_end = w_start + 480
    elif (output_size == 640):
        h_start = max(cand[0]-319, 0)
        w_start = max(cand[1]-319, 0)
        if (h_start+640 > shape_[0]):
            h_start = shape_[0] - 640
        if (w_start+640 > shape_[1]):
            w_start = shape_[1] - 640
        h_end = h_start + 640
        w_end = w_start + 640
    return h_start, h_end, w_start, w_end


def batch_resize_by_scale(img, scale):
    '''
    :param img: The input image, should be shape like [:,:,11]
    :param deter_h: The picture height as you wish to resize to
    :param deter_w: The picture width as you wish to resize to
    :return: A vector with shape [deter_h, deter_w, 11]
    '''
    shape_ = img.shape
    image = np.zeros([shape_[0]*int(scale), shape_[1]*int(scale), 11])
    # try:
    image[:, :, :3] = misc.imresize(img[:, :, :3], scale, interp='nearest').astype(
        np.float64)
    image[:, :, 3] = misc.imresize(img[:, :, 3], scale, interp='nearest').astype(
        np.float64)
    image[:, :, 4] = misc.imresize(img[:, :, 4], scale, interp='nearest').astype(
        np.float64)
    image[:, :, 5:8] = misc.imresize(img[:, :, 5:8], scale, interp='nearest').astype(
        np.float64)
    image[:, :, 8:11] = misc.imresize(img[:, :, 8:11], scale, interp='nearest').astype(
        np.float64)
    return image

def batch_resize(img, deter_h, deter_w):
    '''
    :param img: The input image, should be shape like [:,:,11]
    :param deter_h: The picture height as you wish to resize to
    :param deter_w: The picture width as you wish to resize to
    :return: A vector with shape [deter_h, deter_w, 11]
    '''
    image = np.zeros([deter_h, deter_w, 11])
    # try:
    image[:, :, :3] = misc.imresize(img[:, :, :3].astype(np.uint8), [deter_h, deter_w], interp='nearest').astype(
        np.float64)
    image[:, :, 3] = misc.imresize(img[:, :, 3].astype(np.uint8), [deter_h, deter_w], interp='nearest').astype(
        np.float64)
    image[:, :, 4] = misc.imresize(img[:, :, 4].astype(np.uint8), [deter_h, deter_w], interp='nearest').astype(
        np.float64)
    image[:, :, 5:8] = misc.imresize(img[:, :, 5:8].astype(np.uint8), [deter_h, deter_w], interp='nearest').astype(
        np.float64)
    image[:, :, 8:11] = misc.imresize(img[:, :, 8:11].astype(np.uint8), [deter_h, deter_w], interp='nearest').astype(
        np.float64)
    return image



