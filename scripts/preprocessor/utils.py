import os
import h5py
import cv2
import numpy as np
import random
from scipy import misc as misc
import math

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
    :param trimap: trimap pic
    :return: an index for unknown region
    '''
    index = np.where(img == 128)
    i = random.choice([j for j in range(len(index[0]))])
    return np.array(index)[:, i][:2]

def validUnknownRegion(img, output_size):
    original_shape = np.asarray(img.shape)
    a = b = c = d = -1
    while not (
        (a >= 0 and a < original_shape[0]) and \
        (b >= 0 and b < original_shape[0]) and \
        (c >= 0 and c < original_shape[1]) and \
        (d >= 0 and c < original_shape[1])
        ):   # update candidate until it's valid
        cand = candidateUnknownRegion(img)    
        half = int(math.ceil(output_size / 2))
        # four coners of the image
        a = cand[0] - half
        b = cand[0] + half
        c = cand[1] - half
        d = cand[1] + half
    return half, cand

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
        np.float32)
    image[:, :, 3] = misc.imresize(img[:, :, 3].astype(np.uint8), [deter_h, deter_w], interp='nearest').astype(
        np.float32)
    image[:, :, 4] = misc.imresize(img[:, :, 4].astype(np.uint8), [deter_h, deter_w], interp='nearest').astype(
        np.float32)
    image[:, :, 5:8] = misc.imresize(img[:, :, 5:8].astype(np.uint8), [deter_h, deter_w], interp='nearest').astype(
        np.float32)
    image[:, :, 8:11] = misc.imresize(img[:, :, 8:11].astype(np.uint8), [deter_h, deter_w], interp='nearest').astype(
        np.float32)
    return image




