"""
YOLACT: Real-time Instance Segmentation
Ref: https://arxiv.org/abs/1904.02689

Author: HSU, CHIHCHAO
"""
import tensorflow as tf

from config import build_backbone
from data.anchor import Anchor
from layers.detection import Detect
from layers.fpn import FeaturePyramidNeck
from layers.head import PredictionModule
from layers.protonet import ProtoNet

assert tf.__version__.startswith('2')


class Yolact(tf.keras.Model):
    """YOLACT architecture (ResNet/FPN + ProtoNet + prediction heads)."""

    def __init__(self,
                 backbone,
                 fpn_channels,
                 num_class,
                 num_mask,
                 anchor_params,
                 detect_params):
        super(Yolact, self).__init__()
        base_model, extracted = build_backbone(backbone)

        self.backbone = tf.keras.Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(x).output for x in extracted],
        )
        self.backbone_fpn = FeaturePyramidNeck(fpn_channels)
        self.protonet = ProtoNet(num_mask)

        # semantic segmentation branch (training-only loss; still computed for graph simplicity)
        self.num_seg_classes = num_class - 1
        self.semantic_segmentation = tf.keras.layers.Conv2D(
            self.num_seg_classes, 1, 1, padding='same', name='semantic_segmentation'
        )

        self.anchor_instance = Anchor(**anchor_params)
        priors = self.anchor_instance.get_anchors()

        # Paper Fig. 4: one shared 3x3, then class / box / mask heads
        self.predictionHead = PredictionModule(
            256, len(anchor_params['aspect_ratio']), num_class, num_mask
        )
        self.detect_layer = Detect(anchors=priors, **detect_params)

    def detect(self, pred):
        """Run Fast-NMS detection on model outputs."""
        return self.detect_layer(pred)

    def call(self, inputs, training=None):
        c3, c4, c5 = self.backbone(inputs)
        fpn_out = self.backbone_fpn(c3, c4, c5)

        p3 = fpn_out[0]
        protonet_out = self.protonet(p3)
        # Paper §5: semantic seg loss is train-only (not used at test time)
        if training:
            seg = self.semantic_segmentation(p3)
        else:
            seg = tf.zeros(
                [tf.shape(p3)[0], tf.shape(p3)[1], tf.shape(p3)[2], self.num_seg_classes],
                dtype=protonet_out.dtype,
            )

        pred_cls = []
        pred_offset = []
        pred_mask_coef = []

        for f_map in fpn_out:
            cls, offset, coef = self.predictionHead(f_map)
            pred_cls.append(cls)
            pred_offset.append(offset)
            pred_mask_coef.append(coef)

        return {
            'pred_cls': tf.concat(pred_cls, axis=1),
            'pred_offset': tf.concat(pred_offset, axis=1),
            'pred_mask_coef': tf.concat(pred_mask_coef, axis=1),
            'proto_out': protonet_out,
            'seg': seg,
        }
