# YOLACT Real-time Instance Segmentation
This is a Tensorflow 2.0 implementation of the paper [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689) accepted in ICCV2019. The paper presents a fully-convolutional model for real-instance segmentation based on extending the existing architecture from object detection and its own idea of parallel prototype generation. In this Repo, I focus on reproducing the result by implementing one of the structure "ResNet50-FPN" on MS-COCO datasets. Due to the computational resource limitation, this project is still ongoing. I will keep updating the information here. 

## Model
Here is the illustration of YOLACT from original paper.
![ad](https://github.com/leohsuofnthu/Tensorflow-YOLACT/blob/master/images/model.png)

## Dataset and Pre-processsing
[COCO Dataset](http://cocodataset.org/#download) is used for reproducing the experiment here.

### (1) Download the COCO 2017 Dataset
[2017 Train images](http://images.cocodataset.org/zips/train2017.zip)  / [2017 Val images](http://images.cocodataset.org/zips/val2017.zip) / [2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

### (2) Create TFRecord for training 
In this repo, we convert images and annotations into TFRecord through the */data/create_coco_tfrecord.py.* In this script, I directly resize the image to 550 * 550 and ignore the images with only crowd annotations. Using the following command to create TFRecod.

```bash
python -m  data.create_coco_tfrecord -train_image_dir 'path of train2017' -val_image_dir 'path of val2017' -train_annotations_file 'path of train annotations' -val_annotations_file 'path of val annotations' -output_dir 'path for output TFRecord'
```
## Train
Training procedure can be conducted directly by following command:
```bash
python train.py -tfrecord_dir 'path of TFRecord files'
                -weights 'path to store weights' 
                -train_iter 'number of  iteration for training'
                -batch_size 'batch_size'
                -lr 'learning rate'
                -momentum 'momentum for SGD'
                -weight_decay 'weight_decay rate for SGD'
                -print_interval 'interval for printing training result'
                -save_interval 'interval for conducting validation'
                -valid_iter 'number of iteration for validation'
```
The default hyperparameters in train.py follows the original setting from the paper:
* Batch size = 8, which is recommanded by paper
* SGD optimizer with learning rate 1e-3 and divided by 10 at iterations 280K, 600K, 700K and 750K, using a momentum 0.9, a weight decay 5* 1e-4. In the original implementation of paper, a warm up learning rate 1e-4 and warm up iterations 500 are used, I put all those setting in a learning schedule object in *utils/learning_rate_schedule.py*.
* Random photometrics distortion, horizontal flip(mirroring) and crop are used here for data augmentation.

## Inference (To Be Updated)
## mAP evaluation (To Be Updated)

## Authors

* **HSU, CHIH-CHAO** - *Professional Machine Learning Master Student at [Mila](https://mila.quebec/)* 

## Reference
* https://github.com/dbolya/yolact
