from pycocotools.coco import COCO
from pycocotools import mask

annotations_train = "../data/annotations/instances_train2017.json"
coco_train = COCO(annotations_train)

print(list(coco_train.imgToAnns.keys()))
