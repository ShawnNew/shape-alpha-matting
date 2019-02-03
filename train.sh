#!/bin/bash
LOG=./log/all/train-`date + %Y-%m-%d-%H-%M-%S`.log
solver=./models/solvers/solver_S2.prototxt
weights=./models/snapshots/shapeAlphaNet/ee/Training_iter_260000.caffemodel

caffe train \
	--solver=${solver} \
	--weights=${weights} \
	--gpu=2 2>&1 | tee $LOG
