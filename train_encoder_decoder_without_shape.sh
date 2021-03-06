#!/bin/bash

LOG=./log/without-shape/ee/train-`date +%Y-%m-%d-%H-%M-%S`.log
solver=./models/solvers/solver_without_S1.prototxt
weights=./models/weights/VGG16_SalObjSub.caffemodel

caffe train \
	--solver=${solver} \
	--weights=${weights} \
	--gpu=$1 \
	2>&1 | tee $LOG
