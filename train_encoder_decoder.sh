#!/bin/bash

size=224
DATASET=/home/dl/harddisk/Datasets/adobe/
HDFDATASET=/home/dl/harddisk/Datasets/adobe-h5
LOG=./log/with-shape/ee/train-`date +%Y-%m-%d-%H-%M-%S`.log
solver=./models/solvers/solver_S1.prototxt
VGG=./models/weights/VGG16_SalObjSub.caffemodel
MODEL=./models/snapshots/shapeAlphaNet/
EPOCH=10

for i in `seq 1 $EPOCH`
do
  # ----------------- train one epoch ------------------------
  echo "Training the ${i} epoch..."
  if [ $i -eq 1 ]
  then
    caffe train \
      --solver=${solver} \
      --weights=${VGG} \
      --gpu=$1 \
      2>&1 | tee $LOG
  else
    # -------------- re-generate hdf5 datasets ----------------
    echo "Re-sampling dataset and regenerating..."
    cd ./scripts &&\
    python preprocessing.py \
        -d ${DATASET} \
        -o ${HDFDATASET} \
        -s ${size} &&\
    cd ../

    caffe train \
      --solver=${solver} \
      --weights=${MODEL}$(ls ${MODEL} -t1 | grep .caffemodel | head -n 1) \
      --gpu=$1 \
      2>&1 | tee $LOG
  fi
done

