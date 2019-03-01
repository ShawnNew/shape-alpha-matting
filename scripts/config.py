## Model
shape_model="/home/dl/Codes/shape-alpha-matting/models/shapeAlphaMattingNet_S1_deploy.prototxt" 
_model="/home/dl/Codes/shape-alpha-matting/models/alphaMattingNet_S1_deploy.prototxt" 

## Weights
shape_weights="/home/dl/Codes/shape-alpha-matting/models/snapshots/shapeAlphaNet/Training_iter_29000.caffemodel"
_weights="/home/dl/Codes/shape-alpha-matting/models/snapshots/alphaNet/Training_iter_29000.caffemodel"

## Dataset
source="/home/dl/harddisk/Datasets/adobe/test.txt"

## Input size of deep network
net_input_w = 960
net_input_h = 960

# Unkown area config
unknown_code = 128
