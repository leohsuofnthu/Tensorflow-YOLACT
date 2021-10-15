# YOLACT Real-time Instance Segmentation
## Introduction
This is a Tensorflow 2 implementation of the paper [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689) accepted in ICCV2019. The paper presents a fully-convolutional model for real-instance segmentation based on extending the existing architecture for object detection and its own idea of parallel prototype generation. In this repo, my goal is to provide a general way to use this model, let users have more flexible options (custom dataset, different backbone choice, anchor scale and learning rate schedue) for their own specific need based on idea from original paper.

## Model
Here is the illustration of YOLACT from original paper.
![ad](https://github.com/leohsuofnthu/Tensorflow-YOLACT/blob/master/images/model.png)

## A. Dataset and Pre-processsing

### 1. Prepare the COCO 2017 TFRecord Dataset
[2017 Train images](http://images.cocodataset.org/zips/train2017.zip)  / [2017 Val images](http://images.cocodataset.org/zips/val2017.zip) / [2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) <br/>

Extract the ```/train2017```, ```/val2017```, and ```/annotations/instances_train2017.json```, ```/annotations/instances_val2017.json ```from annotation to ```./data``` folder of the repo, and run:

```bash
python -m  data.coco_tfrecord_creator -train_image_dir './data/train2017' 
                                      -val_image_dir './data/val2017' 
                                      -train_annotations_file './data/instances_train2017.json' 
                                      -val_annotations_file './data/instances_val2017.json' 
                                      -output_dir './data/coco'
```
### 2. Prepare the Pascal SBD Dataset
[benchmark.tgz](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)  /
[Pascal SBD annotation](https://drive.google.com/file/d/1ExrRSPVctHW8Nxrn0SofU1lVhK5Wn0_S/view) (Here is the COCO-style annotation from original yolact repo)  <br/>

Extract the ```/benchmark/dataset/img ``` folder from benchmark.tgz, and ```pascal_sbd_train.json```, ```pascal_sbd_valid.json``` from annotation to ```./data``` folder of the repo. Divinding images into 2 folders (```/pascal_train``` for training, ```/pascal_val``` for validation images.) and run:

```bash
python -m  data.coco_tfrecord_creator -train_image_dir './data/pascal_train' 
                                      -val_image_dir './data/pascal_val' 
                                      -train_annotations_file './data/pascal_sbd_train.json' 
                                      -val_annotations_file './data/pascal_sbd_valid.json' 
                                      -output_dir './data/pascal'
```

### 3. Prepare your Custom Dataset
Create a folder of training images, a folder of validation images, and a COCO-style annotation like above for your dataset in ```./data``` folder of the repo, and run:

```bash
python -m  data.coco_tfrecord_creator -train_image_dir 'path to your training images' 
                                      -val_image_dir   'path to your validaiton images'  
                                      -train_annotations_file 'path to your training annotations' 
                                      -val_annotations_file 'path to your validation annotations' 
                                      -output_dir './data/name of the dataset'
```
## Training
### 1. Configuration for COCO, Pascal SBD
The configuration for experiment can be adjust in ```config.py```. The default hyperparameters from original paper are already written as example for you to know how to customize it. You can adjust following parameters:

#### Parameters for Parser
| Parameters | Description |
| --- | --- |
| `NUM_MAX_PAD` | The maximum padding length for batching samples. |
| `THRESHOLD_POS` | The positive threshold iou for anchor mathcing. |
| `THRESHOLD_NEG` | The negative threshold iou for anchor mathcing. |

#### Parameters for Model
| Parameters | Description |
| --- | --- |
| `BACKBONE` | The name of backbone model defined in `backbones_objects` .|
| `IMG_SIZE` | The input size of images.|
| `PROTO_OUTPUT_SIZE` | Output size of protonet.|
| `FPN_CHANNELS` | The Number of convolution channels used in FPN.|
| `NUM_MASK`| The number of predicted masks for linear combination.|

#### Parameters for Loss
| Parameters for Loss | Description |
| --- | --- |
| `LOSS_WEIGHT_CLS` | The loss weight for classification. |
| `LOSS_WEIGHT_BOX` | The loss weight for bounding box. |
| `LOSS_WEIGHT_MASK` | The loss weight for mask prediction. |
| `LOSS_WEIGHT_SEG` | The loss weight for segamentation. |
| `NEG_POS_RATIO` | The neg/pos ratio for OHEM in classification. |

#### Parameters for Detection
| Parameters | Description |
| --- | --- |
| `CONF_THRESHOLD` | The threshold for filtering possible detection by confidence score. |
| `TOP_K` | The maximum number of input possible detection for FastNMS. |
| `NMS_THRESHOLD` | The threshold for FastNMS. |
| `MAX_NUM_DETECTION` | The maximum number of detection.|


### 2. Configuration for Custom Dataset (to be updated)
```bash




```
### 3. Check the Dataset Sample 
```bash




```

### 4. Training Script
-> Training for COCO:
```bash
python train.py -name 'coco'
                -tfrecord_dir './data'
                -weights './weights' 
                -batch_size '8'
                -momentum '0.9'
                -weight_decay '5 * 1e-4'
                -print_interval '10'
                -save_interval '5000'
```
-> Training for Pascal SBD:
```bash
python train.py -name 'pascal'
                -tfrecord_dir './data'
                -weights './weights' 
                -batch_size '8'
                -momentum '0.9'
                -weight_decay '5 * 1e-4'
                -print_interval '10'
                -save_interval '5000'
```
-> Training for custom dataset:
```bash
python train.py -name 'name of your dataset'
                -tfrecord_dir './data'
                -weights 'path to store weights' 
                -batch_size 'batch_size'
                -momentum 'momentum for SGD'
                -weight_decay 'weight_decay rate for SGD'
                -print_interval 'interval for printing training result'
                -save_interval 'interval for evaluation'
```
## Inference (to be updated)
There are serval evaluation scenario.
```bash




```
### Test Detection
```bash




```
### Evaluation
```bash




```
### Images
```bash




```
### Videos 
```bash




```

## Pretrain Weights (to be updated)
First Header | Second Header
------------ | -------------
Content from cell 1 | Content from cell 2
Content in the first column | Content in the second column
## Authors

* **HSU, CHIH-CHAO** - *Professional Machine Learning Master Student at [Mila](https://mila.quebec/)* 

## Reference
* https://github.com/dbolya/yolact
* https://github.com/leohsuofnthu/Tensorflow-YOLACT/blob/master/data/create_coco_tfrecord.py
* https://github.com/tensorflow/models/blob/master/official/vision/detection/dataloader/retinanet_parser.py
* https://github.com/balancap/SSD-Tensorflow
