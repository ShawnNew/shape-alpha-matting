#!/bin/bash

size=224
DATASET=/home/dl/harddisk/Datasets/adobe/
HDFDATASET=/home/dl/harddisk/Datasets/adobe-h5
LOG=./log/without-shape/ee/train-`date +%Y-%m-%d-%H-%M-%S`.log
solver=./models/solvers/solver_without_S1.prototxt
VGG=./models/weights/VGG16_SalObjSub.caffemodel
MODEL=./models/snapshots/alphaNet/
snapshot=./models/snapshots/alphaNet/Training_iter_14105.solverstate

caffe train \
	--solver=${solver} \
	--snapshot=${MODEL}$(ls ${MODEL} -t1 | grep .solverstate | head -n 1) \
	--gpu=$1 \
	2>&1 | tee $LOG
