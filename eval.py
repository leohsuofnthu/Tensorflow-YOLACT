"""
Mostly adapted from:
"""
import os
from collections import OrderedDict

import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

from utils.APObject import APObject, Detections
from utils.utils import jaccard, mask_iou, postprocess

import config as cfg

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
print(iou_thresholds)


# Todo write test file to check if works correctly
# for calculating IOU between gt and detection box
# so as to decide the TP, FP, FN
def _bbox_iou(bbox1, bbox2):
    ret = jaccard(bbox1, bbox2)
    return ret


# Todo write test file to check if works correctly
# for calculating IOU between gt and detection mask
def _mask_iou(mask1, mask2):
    ret = mask_iou(mask1, mask2)
    return ret


# ref from original arthor
def calc_map(ap_data):
    """

    :param ap_data: all individual AP
    :return:
    """
    print("Calculating mAP...")

    # create empty list of dict for different iou threshold
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    # 　calculate Ap for every classes individually
    for _class in range():
        # each class have multiple different iou threshold to calculate
        for iou_idx in range(len(iou_thresholds)):
            # there are 2 type of mAP we want to know (bounding box and mask)
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                # calculate AP if there is detection in certain class
                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict}

    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

        print_maps(all_maps)

        # Put in a prettier format so we can serialize it to json during training
        all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
        return all_maps


def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


def prep_metrics(ap_data, dets, img, labels, h, w, image_id=None, detections=None):
    """Mainly update the ap_data for validation table"""

    # postprocess the prediction
    # Todo 550 to constant
    classes, scores, boxes, masks = postprocess(dets, 550, 550, 0, "bilinear")
    classes, scores = classes.numpy(), scores.numpy()

    # if no detections
    if classes.shape[0] == 0:
        return

    # prepare gt
    gt_bbox = labels['bbox']
    gt_classes = labels['classes']
    gt_masks = labels['mask_target']

    # prepare data
    classes = list(classes)
    scores = list(scores)
    box_scores = scores
    mask_scores = scores
    """
    why cuda tensor? for iou fast calculation?
    masks = masks.view(-1, h * w).cuda()
    boxes = boxes.cuda()
    """

    # else
    num_pred = len(classes)
    num_gt = len(gt_classes)

    # calculating the IOU first
    mask_iou_cache = _mask_iou(masks, gt_masks)
    bbox_iou_cache = _bbox_iou(boxes, gt_bbox)

    """
    If crowd label included
    if num_crowd > 0:
        crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
        crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
    else:
        crowd_mask_iou_cache = None
        crowd_bbox_iou_cache = None
    """

    ...


def prep_benchmarks():
    ...


def prep_display():
    ...


def eval_image():
    ...


def eval_images():
    ...


def eval_video():
    ...


def evaluate(model, dataset):
    # if use fastnms
    # if use cross class nms

    # if eval image
    # if eval images
    # if eval video

    # if not display or benchmark
    # For mAP evaluation, creating AP_Object for every class per iou_threshold
    ap_data = {
        # Todo add item in config.py
        'box': [[APObject() for _ in cfg.NUM_CLASS] for _ in iou_thresholds],
        'mask': [[APObject() for _ in cfg.NUM_CLASS] for _ in iou_thresholds]}

    # detection object made from prediction output
    detections = Detections()

    # iterate the whole dataset to save TP, FP, FN
    for idx, data in enumerate(dataset):
        preds = model(...)
        prep_metrics(ap_data, preds, img, gt, gt_masks, w, h, detections)

    # if to json
    # save detection to json

    # Todo if not training, save ap_data, else calc_map
    return calc_map(ap_data)


def main(argv):
    pass


if __name__ == '__main__':
    app.run(main)
