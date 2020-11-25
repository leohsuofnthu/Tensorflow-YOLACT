# YOLACT Real-time Instance Segmentation
This is a Tensorflow 2.0 implementation of the paper [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689) accepted in ICCV2019. The paper presents a fully-convolutional model for real-instance segmentation based on extending the existing architecture for object detection and its own idea of parallel prototype generation. In this repo, my goal is to provide a general way to use this model, let users have more flexible options (custom dataset, different backbone choice, anchor scale and learning rate schedue) for their own specific need based on idea from original paper.

## Model
Here is the illustration of YOLACT from original paper.
![ad](https://github.com/leohsuofnthu/Tensorflow-YOLACT/blob/master/images/model.png)

## Dataset and Pre-processsing

### Prepare the COCO 2017 TFRecord Dataset
[2017 Train images](http://images.cocodataset.org/zips/train2017.zip)  / [2017 Val images](http://images.cocodataset.org/zips/val2017.zip) / [2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) <br/>

Extract the ```/train2017```, ```/val2017```, and ```/annotations/instances_train2017.json```, ```/annotations/instances_val2017.json ```from annotation to ```./data``` folder of the repo, and run:

```bash
python -m  data.coco_tfrecord_creator -train_image_dir './data/train2017' 
                                      -val_image_dir './data/val2017' 
                                      -train_annotations_file './data/instances_train2017.json' 
                                      -val_annotations_file './instances_val2017.json' 
                                      -output_dir './data/coco'
```
### Prepare the Pascal SBD Dataset
[benchmark.tgz](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)  /
[Pascal SBD annotation](https://drive.google.com/file/d/1ExrRSPVctHW8Nxrn0SofU1lVhK5Wn0_S/view) (Here is the COCO-style annotation from original yolact repo)  <br/>

Extract the ```/benchmark/dataset/img ``` folder from benchmark.tgz, and ```pascal_sbd_train.json```, ```pascal_sbd_valid.json``` from annotation to ```./data``` folder of the repo. Divinding images into 2 folders (```/pascal_train``` for training, ```/pascal_val``` for validation images.) and run:

```bash
python -m  data.coco_tfrecord_creator -train_image_dir './data/pascal_train' 
                                      -val_image_dir './data/pascal_val' 
                                      -train_annotations_file './data/pascal_sbd_train.json' 
                                      -val_annotations_file './pascal_sbd_valid.json' 
                                      -output_dir './data/pascal'
```

### Prepare your Custom Dataset
Create a folder of training images, a folder of validation images, and a COCO-style annotation like above for your dataset in ```./data``` folder of the repo, and run:

```bash
python -m  data.coco_tfrecord_creator -train_image_dir 'path to your training images' 
                                      -val_image_dir   'path to your validaiton images'  
                                      -train_annotations_file 'path to your training annotations' 
                                      -val_annotations_file 'path to your validation annotations' 
                                      -output_dir './data/name of the dataset'
```

### Check and Load the Dataset
```bash




```
## Training
### Configuration

### Training Script
Training procedure can be conducted directly by following command:
```bash
python train.py -name 'coco'
                -tfrecord_dir 'path of TFRecord files'
                -weights 'path to store weights' 
                -batch_size 'batch_size'
                -momentum 'momentum for SGD'
                -weight_decay 'weight_decay rate for SGD'
                -print_interval 'interval for printing training result'
                -save_interval 'interval for conducting validation'
```
The default hyperparameters in train.py follows the original setting from the paper:
* Batch size = 8, which is recommanded by paper
* SGD optimizer with learning rate 1e-3 and divided by 10 at iterations 280K, 600K, 700K and 750K, using a momentum 0.9, a weight decay 5* 1e-4. In the original implementation of paper, a warm up learning rate 1e-4 and warm up iterations 500 are used, I put all those setting in a learning schedule object in *utils/learning_rate_schedule.py*.
* Random photometrics distortion, horizontal flip(mirroring) and crop are used here for data augmentation.


## Inference 
### Evaluation
### Images [Soon]
### Videos [Soon]

## Pretrain Weights [Soon]

## Authors

* **HSU, CHIH-CHAO** - *Professional Machine Learning Master Student at [Mila](https://mila.quebec/)* 

## Reference
* https://github.com/dbolya/yolact
* https://github.com/leohsuofnthu/Tensorflow-YOLACT/blob/master/data/create_coco_tfrecord.py
* https://github.com/tensorflow/models/blob/master/official/vision/detection/dataloader/retinanet_parser.py
* https://github.com/balancap/SSD-Tensorflow
