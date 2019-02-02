#!/bin/bash
LOG=./log/all/train-`date + %Y-%m-%d-%H-%M-%S`.log
solver=./models/shapeAlphaMattingNet_S2.prototxt
weights=./models/snapshots/shapeAlphaNet/ee/

caffe train \
	--solver=${solver} \
	--weights=${weights} \
	--gpu=2 2>&1 | tee $LOG
