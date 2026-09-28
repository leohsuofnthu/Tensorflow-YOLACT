"""
Evaluation and inference for Tensorflow-YOLACT.

Examples:
  # Validate mAP on TFRecords
  python eval.py --mode val --name coco --tfrecord_dir ./data --weights ./weights/weights_coco.h5

  # Run on a single image
  python eval.py --mode image --image ./demo.jpg --weights ./weights/weights_coco.h5 --output_dir ./results

  # Run on a folder of images
  python eval.py --mode images --image_dir ./demo_images --weights ./weights/weights_coco.h5 --output_dir ./results

  # Run on a video
  python eval.py --mode video --video ./demo.mp4 --weights ./weights/weights_coco.h5 --output_dir ./results
"""
import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from tensorflow.keras.utils import Progbar

from config import IMG_SIZE, LABEL_MAP, get_params
from data.coco_dataset import ObjectDetectionDataset
from utils.APObject import APObject, Detections
from utils.utils import jaccard, mask_iou, postprocess
from utils.visualize import draw_detections, get_class_names
from yolact import Yolact

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'image', ['val', 'image', 'images', 'video'],
                  'Evaluation / inference mode')
flags.DEFINE_string('name', 'coco', 'Dataset name (config key)')
flags.DEFINE_string('tfrecord_dir', './data', 'Parent directory of TFRecords')
flags.DEFINE_string('weights', '', 'Path to .h5 weights or a checkpoint directory')
flags.DEFINE_string('image', '', 'Path to a single image')
flags.DEFINE_string('image_dir', '', 'Directory of images')
flags.DEFINE_string('video', '', 'Path to a video file')
flags.DEFINE_string('output_dir', './results', 'Where to write visualized outputs')
flags.DEFINE_float('score_threshold', 0.3, 'Score threshold for display / metrics')
flags.DEFINE_bool('display', False, 'Show a window while processing (image/video)')
flags.DEFINE_string('bbox_det_file', './results/bbox_detections.json',
                    'COCO bbox JSON path (val mode)')
flags.DEFINE_string('mask_det_file', './results/mask_detections.json',
                    'COCO mask JSON path (val mode)')
flags.DEFINE_bool('output_coco_json', False, 'Dump COCO JSON during val')

iou_thresholds = [x / 100 for x in range(50, 100, 5)]


def _bbox_iou(bbox1, bbox2, is_crowd=False):
    return jaccard(bbox1, bbox2, is_crowd)


def _mask_iou(mask1, mask2, is_crowd=False):
    return mask_iou(mask1, mask2, is_crowd)


def calc_map(ap_data, num_cls):
    tf.print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(num_cls):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]
                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0
        for i, threshold in enumerate(iou_thresholds):
            vals = aps[i][iou_type]
            mAP = sum(vals) / len(vals) * 100 if vals else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (
            sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1)
        )

    print_maps(all_maps)
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


def print_maps(all_maps):
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)

    tf.print()
    tf.print(make_row([''] + [
        ('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()
    ]))
    tf.print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        tf.print(make_row([iou_type] + [
            '%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()
        ]))
    tf.print(make_sep(len(all_maps['box']) + 1))
    tf.print()


def prep_metrics(ap_data, dets, img, labels, detections=None, image_id=None,
                 score_threshold=0.05, output_coco_json=False):
    """Update ap_data (and optionally detections) for one validation batch."""
    # img is NHWC; for square YOLACT inputs H == W
    h = int(tf.shape(img)[1])
    w = int(tf.shape(img)[2])

    classes, scores, boxes, masks = postprocess(
        dets, w, h, 0, 'bilinear', score_threshold=score_threshold
    )
    if classes is None:
        return

    if tf.size(scores) == 1:
        scores = tf.expand_dims(scores, axis=0)
        masks = tf.expand_dims(masks, axis=0)
    boxes = tf.expand_dims(boxes, axis=0) if tf.rank(boxes) == 1 else boxes
    if tf.rank(boxes) == 2:
        boxes = tf.expand_dims(boxes, axis=0)

    gt_bbox = labels['bbox']
    gt_classes = labels['classes']
    gt_masks = labels['mask_target']
    num_obj = labels['num_obj']
    num_gt = int(num_obj.numpy()[0])

    classes_np = list(classes.numpy())
    scores_np = list(scores.numpy())
    box_scores = scores_np
    mask_scores = scores_np
    num_pred = len(classes_np)

    if output_coco_json and detections is not None:
        boxes_np = boxes.numpy()
        if boxes_np.ndim == 3:
            boxes_np = boxes_np[0]
        masks_np = masks.numpy()
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        for i in range(masks_np.shape[0]):
            x1, y1, x2, y2 = boxes_np[i]
            if (y2 - y1) * (x2 - x1) > 0:
                detections.add_box(image_id, classes_np[i], boxes_np[i], box_scores[i])
                detections.add_mask(image_id, classes_np[i], masks_np[i], mask_scores[i])
        return

    masks_gt = tf.squeeze(
        tf.image.resize(tf.expand_dims(gt_masks[0], axis=-1), [h, w], method='bilinear'),
        axis=-1,
    )

    mask_iou_cache = _mask_iou(masks, masks_gt).numpy()
    bbox_iou_cache = tf.squeeze(_bbox_iou(boxes, gt_bbox), axis=0).numpy()

    box_indices = sorted(range(num_pred), key=lambda idx: -box_scores[idx])
    mask_indices = sorted(box_indices, key=lambda idx: -mask_scores[idx])

    iou_types = [
        ('box', lambda row, col: bbox_iou_cache[row, col],
         lambda idx: box_scores[idx], box_indices),
        ('mask', lambda row, col: mask_iou_cache[row, col],
         lambda idx: mask_scores[idx], mask_indices),
    ]

    # model uses background=0; convert to 0-based foreground class indices
    gt_classes_list = list(gt_classes[0].numpy()[:num_gt] - 1)

    for _class in set(classes_np + gt_classes_list):
        num_gt_for_class = sum(1 for x in gt_classes_list if x == _class)
        for iou_idx, th in enumerate(iou_thresholds):
            for iou_type, iou_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes_list)
                ap_obj = ap_data[iou_type][iou_idx][_class]
                ap_obj.add_gt_positive(num_gt_for_class)

                for i in indices:
                    if classes_np[i] != _class:
                        continue
                    max_iou_found = th
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes_list[j] != _class:
                            continue
                        iou = iou_func(i, j)
                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        ap_obj.push(score_func(i), False)


def evaluate(model, dataset, num_val, num_cls, score_threshold=0.05,
             output_coco_json=False, detections=None):
    ap_data = {
        'box': [[APObject() for _ in range(num_cls)] for _ in iou_thresholds],
        'mask': [[APObject() for _ in range(num_cls)] for _ in iou_thresholds],
    }
    if detections is None:
        detections = Detections(label_map=LABEL_MAP.get(FLAGS.name))

    progbar = Progbar(num_val)
    logging.info('Evaluating...')
    for i, (image, labels) in enumerate(dataset):
        output = model(image, training=False)
        dets = model.detect(output)
        image_id = i
        prep_metrics(
            ap_data, dets, image, labels, detections,
            image_id=image_id,
            score_threshold=score_threshold,
            output_coco_json=output_coco_json,
        )
        progbar.update(i + 1)

    if output_coco_json:
        os.makedirs(os.path.dirname(FLAGS.bbox_det_file) or '.', exist_ok=True)
        detections.to_json(FLAGS.bbox_det_file, FLAGS.mask_det_file)
        logging.info('Wrote COCO JSON to %s and %s', FLAGS.bbox_det_file, FLAGS.mask_det_file)
        return None

    return calc_map(ap_data, num_cls)


def _build_model(dataset_name):
    _, _, _, _, _, _, model_params = get_params(dataset_name)
    model = Yolact(**model_params)
    # Build variables with a dummy forward pass
    dummy = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
    _ = model(dummy, training=False)
    return model


def _load_weights(model, weights_path):
    if not weights_path:
        raise ValueError('Please pass --weights /path/to/file.weights.h5 (or a checkpoint dir).')
    if os.path.isdir(weights_path):
        ckpt = tf.train.Checkpoint(model=model)
        latest = tf.train.latest_checkpoint(weights_path)
        if latest is None:
            raise FileNotFoundError(f'No checkpoint found in {weights_path}')
        ckpt.restore(latest).expect_partial()
        logging.info('Restored checkpoint %s', latest)
    else:
        # Keras 3 prefers *.weights.h5; still accept older *.h5 if present
        try:
            model.load_weights(weights_path)
        except ValueError:
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        logging.info('Loaded weights from %s', weights_path)


def _preprocess_bgr(image_bgr, dataset_name):
    """Return (network_input[1,H,W,3], orig_rgb, scale helpers)."""
    _, _, _, _, _, parser_params, model_params = get_params(dataset_name)
    preprocess = parser_params['augmentation_params']['preprocess_func']
    img_size = IMG_SIZE

    orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_rgb.shape[:2]
    resized = cv2.resize(orig_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    inp = preprocess(resized.astype(np.float32))
    inp = np.expand_dims(inp, axis=0)
    return inp, orig_rgb, orig_h, orig_w, img_size


def _detect_numpy(model, inp, score_threshold):
    output = model(inp, training=False)
    dets = model.detect(output)
    classes, scores, boxes, masks = postprocess(
        dets, IMG_SIZE, IMG_SIZE, 0, 'bilinear', score_threshold=score_threshold
    )
    if classes is None:
        return None, None, None, None

    classes = classes.numpy()
    scores = scores.numpy()
    boxes = boxes.numpy()
    masks = masks.numpy()
    if masks.ndim == 2:
        masks = masks[None, ...]
    if boxes.ndim == 1:
        boxes = boxes[None, ...]
    return classes, scores, boxes, masks


def _scale_to_original(boxes, masks, orig_h, orig_w, img_size):
    scale_x = orig_w / float(img_size)
    scale_y = orig_h / float(img_size)
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    resized_masks = []
    for m in masks:
        resized_masks.append(
            cv2.resize(m.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        )
    return boxes, np.stack(resized_masks, axis=0)


def prep_display(image_bgr, classes, scores, boxes, masks, class_names, score_threshold):
    return draw_detections(
        image_bgr, classes, scores, boxes, masks, class_names,
        score_threshold=score_threshold,
    )


def eval_image(model, image_path, output_dir, dataset_name, score_threshold, display=False):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    inp, _, orig_h, orig_w, img_size = _preprocess_bgr(image_bgr, dataset_name)
    classes, scores, boxes, masks = _detect_numpy(model, inp, score_threshold)
    class_names = get_class_names(dataset_name)

    if classes is None:
        result = image_bgr
        logging.info('No detections above threshold for %s', image_path)
    else:
        boxes, masks = _scale_to_original(boxes, masks, orig_h, orig_w, img_size)
        result = prep_display(image_bgr, classes, scores, boxes, masks,
                              class_names, score_threshold)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, result)
    logging.info('Wrote %s', out_path)

    if display:
        cv2.imshow('YOLACT', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out_path


def eval_images(model, image_dir, output_dir, dataset_name, score_threshold, display=False):
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
                '*.JPG', '*.JPEG', '*.PNG']
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(image_dir, p)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f'No images found in {image_dir}')

    logging.info('Running on %d images...', len(paths))
    for path in paths:
        eval_image(model, path, output_dir, dataset_name, score_threshold, display=False)
        if display:
            shown = cv2.imread(os.path.join(output_dir, os.path.basename(path)))
            cv2.imshow('YOLACT', shown)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if display:
        cv2.destroyAllWindows()


def eval_video(model, video_path, output_dir, dataset_name, score_threshold, display=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {video_path}')

    os.makedirs(output_dir, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + '_yolact.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    class_names = get_class_names(dataset_name)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        inp, _, orig_h, orig_w, img_size = _preprocess_bgr(frame, dataset_name)
        classes, scores, boxes, masks = _detect_numpy(model, inp, score_threshold)
        if classes is None:
            result = frame
        else:
            boxes, masks = _scale_to_original(boxes, masks, orig_h, orig_w, img_size)
            result = prep_display(frame, classes, scores, boxes, masks,
                                  class_names, score_threshold)
        writer.write(result)
        frame_idx += 1
        if frame_idx % 30 == 0:
            logging.info('Processed %d frames...', frame_idx)
        if display:
            cv2.imshow('YOLACT', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()
    logging.info('Wrote %s', out_path)
    return out_path


def main(argv):
    del argv
    model = _build_model(FLAGS.name)
    _load_weights(model, FLAGS.weights)

    if FLAGS.mode == 'val':
        _, _, num_cls, _, _, parser_params, model_params = get_params(FLAGS.name)
        # Rebuild so anchors match dataset config used by dataloader
        model = Yolact(**model_params)
        dummy = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
        _ = model(dummy, training=False)
        _load_weights(model, FLAGS.weights)

        dataset = ObjectDetectionDataset(
            dataset_name=FLAGS.name,
            tfrecord_dir=os.path.join(FLAGS.tfrecord_dir, FLAGS.name),
            anchor_instance=model.anchor_instance,
            **parser_params,
        )
        valid_dataset = dataset.get_dataloader(subset='val', batch_size=1)
        num_val = sum(1 for _ in valid_dataset)
        valid_dataset = dataset.get_dataloader(subset='val', batch_size=1)

        evaluate(
            model, valid_dataset, num_val, num_cls,
            score_threshold=FLAGS.score_threshold,
            output_coco_json=FLAGS.output_coco_json,
        )
    elif FLAGS.mode == 'image':
        if not FLAGS.image:
            raise ValueError('--image is required for mode=image')
        eval_image(model, FLAGS.image, FLAGS.output_dir, FLAGS.name,
                   FLAGS.score_threshold, FLAGS.display)
    elif FLAGS.mode == 'images':
        if not FLAGS.image_dir:
            raise ValueError('--image_dir is required for mode=images')
        eval_images(model, FLAGS.image_dir, FLAGS.output_dir, FLAGS.name,
                    FLAGS.score_threshold, FLAGS.display)
    elif FLAGS.mode == 'video':
        if not FLAGS.video:
            raise ValueError('--video is required for mode=video')
        eval_video(model, FLAGS.video, FLAGS.output_dir, FLAGS.name,
                   FLAGS.score_threshold, FLAGS.display)


if __name__ == '__main__':
    app.run(main)
