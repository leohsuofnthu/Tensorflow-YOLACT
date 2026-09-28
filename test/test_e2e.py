"""
End-to-end tests for Tensorflow-YOLACT (no pretrained weights required).

Run from repo root:
  python -m pytest test/test_e2e.py -v
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

# Ensure repo root is on path when pytest collects from test/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import IMG_SIZE, NUM_MASK, get_params
from loss.loss_yolact import YOLACTLoss
from utils.utils import postprocess
from utils.visualize import get_class_names
from yolact import Yolact


@pytest.fixture(scope='module')
def coco_model():
    _, _, _, _, _, _, model_params = get_params('coco')
    model = Yolact(**model_params)
    _ = model(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]), training=False)
    return model


@pytest.fixture(scope='module')
def loss_params():
    _, _, _, _, loss_params, _, _ = get_params('coco')
    return loss_params


class TestArchitecture:
    def test_forward_shapes(self, coco_model):
        x = tf.random.uniform([2, IMG_SIZE, IMG_SIZE, 3])
        out = coco_model(x, training=True)
        assert out['pred_cls'].shape[0] == 2
        assert out['pred_cls'].shape[-1] == 81
        assert out['pred_offset'].shape[-1] == 4
        assert out['pred_mask_coef'].shape[-1] == NUM_MASK
        assert tuple(out['proto_out'].shape[1:3]) == (138, 138)
        assert out['proto_out'].shape[-1] == NUM_MASK
        assert out['seg'].shape[-1] == 80  # num_class - 1
        # FPN produces 5 levels -> 19248 anchors for 550 with 3 ratios
        assert out['pred_cls'].shape[1] == 19248

    def test_seg_skipped_at_inference(self, coco_model):
        x = tf.random.uniform([1, IMG_SIZE, IMG_SIZE, 3])
        out = coco_model(x, training=False)
        assert tf.reduce_sum(tf.abs(out['seg'])) == 0

    def test_mask_coeffs_tanh_bounded(self, coco_model):
        x = tf.random.uniform([1, IMG_SIZE, IMG_SIZE, 3])
        out = coco_model(x, training=False)
        coef = out['pred_mask_coef']
        assert tf.reduce_max(coef) <= 1.0 + 1e-5
        assert tf.reduce_min(coef) >= -1.0 - 1e-5

    def test_detect_returns_batch(self, coco_model):
        x = tf.random.uniform([1, IMG_SIZE, IMG_SIZE, 3])
        out = coco_model(x, training=False)
        dets = coco_model.detect(out)
        assert isinstance(dets, list)
        assert len(dets) == 1
        assert 'detection' in dets[0]


class TestWeightsRoundtrip:
    def test_save_load_weights(self, coco_model, tmp_path):
        path = str(tmp_path / 'yolact.weights.h5')
        x = tf.random.uniform([1, IMG_SIZE, IMG_SIZE, 3], seed=0)
        out1 = coco_model(x, training=False)
        coco_model.save_weights(path)

        _, _, _, _, _, _, model_params = get_params('coco')
        model2 = Yolact(**model_params)
        _ = model2(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]), training=False)
        model2.load_weights(path)
        out2 = model2(x, training=False)

        for key in ('pred_cls', 'pred_offset', 'pred_mask_coef', 'proto_out'):
            np.testing.assert_allclose(
                out1[key].numpy(), out2[key].numpy(), rtol=1e-5, atol=1e-5
            )


class TestInferencePath:
    def test_eval_helpers_on_synthetic_image(self, coco_model, tmp_path):
        import cv2
        from eval import (
            _detect_numpy,
            _preprocess_bgr,
            _scale_to_original,
            prep_display,
        )

        # Synthetic BGR image (non-square to exercise H/W scaling)
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:] = (30, 60, 90)
        cv2.rectangle(img, (40, 30), (200, 180), (0, 200, 100), -1)
        img_path = str(tmp_path / 'synth.jpg')
        cv2.imwrite(img_path, img)

        image_bgr = cv2.imread(img_path)
        inp, _, orig_h, orig_w, img_size = _preprocess_bgr(image_bgr, 'coco')
        assert inp.shape == (1, IMG_SIZE, IMG_SIZE, 3)
        assert (orig_h, orig_w) == (240, 320)

        classes, scores, boxes, masks = _detect_numpy(
            coco_model, inp, score_threshold=0.001
        )
        # Random weights may yield zero detections; both outcomes are valid
        if classes is None:
            result = image_bgr
        else:
            boxes, masks = _scale_to_original(boxes, masks, orig_h, orig_w, img_size)
            assert boxes.shape[-1] == 4
            assert masks.shape[1:] == (orig_h, orig_w)
            result = prep_display(
                image_bgr, classes, scores, boxes, masks,
                get_class_names('coco'), score_threshold=0.001,
            )

        out_path = str(tmp_path / 'out.jpg')
        assert cv2.imwrite(out_path, result)
        assert os.path.isfile(out_path)
        assert result.shape == image_bgr.shape

    def test_postprocess_resize_uses_height_width(self, coco_model):
        """Masks must be resized to [height, width], not [width, height]."""
        x = tf.random.uniform([1, IMG_SIZE, IMG_SIZE, 3])
        out = coco_model(x, training=False)
        # Force a detection dict shaped like Detect output with one fake det
        proto = out['proto_out'][0]
        fake = [{
            'detection': {
                'box': tf.constant([[10.0, 20.0, 100.0, 120.0]], dtype=tf.float32),
                'mask': out['pred_mask_coef'][0, :1],
                'class': tf.constant([0], dtype=tf.int32),
                'score': tf.constant([0.99], dtype=tf.float32),
                'proto': proto,
            }
        }]
        # Non-square target size
        height, width = 200, 300
        classes, scores, boxes, masks = postprocess(
            fake, width, height, 0, score_threshold=0.5
        )
        assert classes is not None
        assert tuple(masks.shape) == (height, width) or tuple(masks.shape) == (1, height, width)


class TestLossStep:
    def test_loss_finite_on_synthetic_batch(self, coco_model, loss_params):
        """One forward + loss with synthetic matched labels (no TFRecord)."""
        batch = 1
        num_anchors = 19248
        num_cls = 81
        num_obj = 2
        pos_idx = np.array([10, 20, 30, 40], dtype=np.int32)

        x = tf.random.uniform([batch, IMG_SIZE, IMG_SIZE, 3])
        pred = coco_model(x, training=True)

        positiveness = np.zeros([batch, num_anchors], dtype=np.float32)
        positiveness[0, pos_idx] = 1.0
        cls_targets = np.zeros([batch, num_anchors], dtype=np.float32)
        cls_targets[0, pos_idx] = 1.0  # class 1
        box_targets = np.zeros([batch, num_anchors, 4], dtype=np.float32)
        max_id = np.zeros([batch, num_anchors], dtype=np.int32)
        max_gt = np.zeros([batch, num_anchors, 4], dtype=np.float32)
        max_gt[0, pos_idx] = np.array([20.0, 20.0, 80.0, 80.0], dtype=np.float32)

        mask_target = (np.random.rand(batch, 100, 138, 138) > 0.7).astype(np.float32)
        classes = np.zeros([batch, 100], dtype=np.float32)
        classes[0, 0] = 1.0
        classes[0, 1] = 2.0

        labels = {
            'cls_targets': tf.constant(cls_targets),
            'box_targets': tf.constant(box_targets),
            'positiveness': tf.constant(positiveness),
            'mask_target': tf.constant(mask_target),
            'max_id_for_anchors': tf.constant(max_id),
            'max_gt_for_anchors': tf.constant(max_gt),
            'classes': tf.constant(classes),
            'num_obj': tf.constant([num_obj], dtype=tf.int32),
        }

        criterion = YOLACTLoss(**loss_params)
        loc, conf, mask, seg, total = criterion(pred, labels, num_cls)
        for t in (loc, conf, mask, seg, total):
            assert tf.math.is_finite(t)
        assert float(total) >= 0.0
