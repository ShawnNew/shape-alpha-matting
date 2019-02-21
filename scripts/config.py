## Model
shape_model="/home/dl/Codes/shape-alpha-matting/models/shapeAlphaMattingNet_deploy.prototxt" 
_model="/home/dl/Codes/shape-alpha-matting/models/alphaMattingNet_deploy.prototxt" 

## Weights
shape_weights="/home/dl/Codes/shape-alpha-matting/models/snapshots/shapeAlphaNet/all/Refine_training_iter_1000.caffemodel"
_weights="/home/dl/Codes/shape-alpha-matting/models/snapshots/alphaNet/all/Refine_training_iter_13500.caffemodel"

## Dataset
source="/home/dl/harddisk/Datasets/adobe/test.txt"

## Input size of deep network
net_input_w = 896
net_input_h = 896

# Unkown area config
unknown_code = 128
