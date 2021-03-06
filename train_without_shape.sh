#!/bin/bash
LOG=./log/without-shape/all/train-`date +%Y-%m-%d-%H-%M-%S`.log
solver=./models/solvers/solver_without_S2.prototxt
weights=./models/snapshots/alphaNet/ee/Training_iter_200000.caffemodel

caffe train \
	--solver=${solver} \
	--weights=${weights} \
	--gpu=$1 2>&1 | tee $LOG
