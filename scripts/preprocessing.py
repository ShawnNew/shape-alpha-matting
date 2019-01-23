import argparse
import pdb
from preprocessor import *

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
    dataset_root = args.directory
    output_dir = args.output
    img_size = args.size
    # SAMPLES = 256                 # samples each hdf5 file
    
    preprocessor = Preprocessor(dataset_root, output_dir, img_size)
    preprocessor.splitDataset(prop_train=0.7, \
                            prop_test=0.15, prop_val=0.15, shuffle=True)
    preprocessor.writeHDF5Files()
