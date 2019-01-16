---
name: VGG16
caffemodel: VGG16_SalObjSub.caffemodel
caffemodel_url: http://www.cs.bu.edu/groups/ivc/data/SOS/VGG16_SalObjSub.caffemodel
license: http://creativecommons.org/licenses/by-nc/4.0/ (non-commercial use only)
caffe_version: trained using a custom Caffe-based framework (see https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md)
gist_id: 27c1c0a7736ba66c2395
---

## Description

The model is used to tackle the alpha matting problem. It is basically an encoder-decoder deep neural network. By feed the network an original image with tri-map, you can get the prediction alpha of the image.


    Deep Image Matting
    Ning Xu, Brian Price, Scott Cohen, and Thomas Huang. 
    CVPR, 2017.

The input images should be mean pixel subtraction. And the channel should be BGR.

## Caffe compatibility

see https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

## How to use the model

The model outputs an array of the input test image, and the array can be converted into an image. In our implementations, test images are resized to 224*224, regardless of the original aspect ratios. 
