"""
Mostly adapted from: https://github.com/dbolya/yolact/blob/master/eval.py
"""
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

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
    for _class in range(cfg.NUM_CLASS):
        # each class have multiple different iou threshold to calculate
        for iou_idx in range(len(iou_thresholds)):
            # there are 2 type of mAP we want to know (bounding box and mask)
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                # calculate AP if there is detection in certain class
                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

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
    # Todo Bug when detection is 1 only
    classes = list(classes)
    scores = list(scores)
    box_scores = scores
    mask_scores = scores

    """
    why cuda tensor? for iou fast calculation?
    masks = masks.view(-1, h * w).cuda()
    boxes = boxes.cuda()
    """
    # if output json, add things to detections objects

    # else
    num_pred = len(classes)
    num_gt = len(gt_classes)

    tf.print(num_pred)
    tf.print(num_gt)

    # resize gt mask
    masks_gt = tf.squeeze(tf.image.resize(tf.expand_dims(gt_masks[0][:num_gt], axis=-1), [550, 550],
                                          method='bilinear'), axis=-1)

    # calculating the IOU first
    mask_iou_cache = _mask_iou(masks, masks_gt).numpy()
    bbox_iou_cache = _bbox_iou(boxes, gt_bbox[0][:num_gt]).numpy()

    tf.print(mask_iou_cache)
    tf.print(bbox_iou_cache)

    """
    # If crowd label included, split it and calculate iou separately from non-crowd label
    if num_crowd > 0:
        crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
        crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
    else:
        crowd_mask_iou_cache = None
        crowd_bbox_iou_cache = None
    """

    # get the sorted index of scores (descending order)
    box_indices = sorted(range(num_pred), key=lambda idx: -box_scores[idx])
    mask_indices = sorted(box_indices, key=lambda idx: -mask_scores[idx])

    # define some useful lambda function for next section
    # avoid writing "bbox_iou_cache[row, col]" too many times, wrap it as a lambda func
    iou_types = [
        ('box', lambda row, col: bbox_iou_cache[row, col].item(),
         # lambda i, j: crowd_bbox_iou_cache[i, j].item(),
         lambda idx: box_scores[idx], box_indices),
        ('mask', lambda row, col: mask_iou_cache[row, col].item(),
         # lambda i,j: crowd_mask_iou_cache[i,j].item(),
         lambda idx: mask_scores[idx], mask_indices)
    ]

    gt_classes = list(gt_classes[0][:num_gt].numpy())

    # starting to update the ap_data from this batch
    for _class in set(classes + gt_classes):
        # calculating how many labels belong to this class
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx in range(len(iou_thresholds)):
            th = iou_thresholds[iouIdx]

            for iou_type, iou_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)

                # get certain APobject
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positive(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue

                    max_iou_found = th
                    max_match_idx = -1

                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                        iou = iou_func(i, j)
                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        matched_crowd = False
                        # for crowd annotation, if no, push as false positive
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)


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


def evaluate(model, detection_layer, dataset, batch_size=1):
    # if use fastnms
    # if use cross class nms

    # if eval image
    # if eval images
    # if eval video

    # if not display or benchmark
    # For mAP evaluation, creating AP_Object for every class per iou_threshold
    ap_data = {
        # Todo add item in config.py
        'box': [[APObject() for _ in range(cfg.NUM_CLASS)] for _ in iou_thresholds],
        'mask': [[APObject() for _ in range(cfg.NUM_CLASS)] for _ in iou_thresholds]}

    # detection object made from prediction output
    detections = Detections()

    # pb = Progbar(1000)
    # iterate the whole dataset to save TP, FP, FN
    i = 0
    for image, labels in dataset:
        tf.print(i)
        i += 1
        output = model(image, training=False)
        detection = detection_layer(output)
        # update ap_data or detection depends if u want to save it to json or just for validation table
        # Todo 550 to variable
        prep_metrics(ap_data, detection, image, labels, 550, 550, detections)
        # pb.add(batch_size)
        if i == 10:
            break

    # if to json
    # save detection to json

    # Todo if not training, save ap_data, else calc_map
    return calc_map(ap_data)


def main(argv):
    pass


if __name__ == '__main__':
    app.run(main)
