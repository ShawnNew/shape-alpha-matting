import os
import h5py
import cv2
import numpy as np
import random

def writeH5Files(out_dir, data, tri_map, gt, fg, bg, file):
    """
    Write the formatted data into hdf5 file
    """
    # h5_filedir = os.path.join(out_dir, h5_filename)
    hdf_file = h5py.File(file, 'w')
    data_shape = data[0].shape
    greyscale_shape = tri_map[0].shape
    chunks_data = (1, data_shape[0], data_shape[1], data_shape[2])
    chunks_greyscale = ( 1, greyscale_shape[0], \
                        greyscale_shape[1],
                        greyscale_shape[2])
    hdf_file.create_dataset('data', dtype=np.float, data=data, chunks=chunks_data)
    hdf_file.create_dataset('tri-map', dtype=np.float, data=tri_map, chunks=chunks_greyscale)
    hdf_file.create_dataset('gt', dtype=np.float, data=gt, chunks=chunks_greyscale)
    hdf_file.create_dataset('fg', dtype=np.float, data=fg, chunks=chunks_data)
    hdf_file.create_dataset('bg', dtype=np.float, data=bg, chunks=chunks_data)
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


def caffeTransform(img, size):
    """
    Transform the data into caffe capable format.
    """
    img = cv2.resize(img, \
                    (size, size),\
                    interpolation=cv2.INTER_CUBIC
                )
    if len(img.shape) == 3: # 3-D image like original image
        img = np.transpose(img, (2, 0, 1))
    else:                   # tri-map and ground truth
        pass
    #TODO: normalization to be done here
    return img / 255.

def greyscaleTransform(img):
    """
    BGR image to greyscale image transformation.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def unknownRegion(trimap):
    '''
    :param trimap: trimap pic
    :return: an index for unknown region
    '''
    index = np.where(trimap == 128)
    i = random.choice([j for j in range(len(index[0]))])
    return np.array(index)[:, i][:2]

