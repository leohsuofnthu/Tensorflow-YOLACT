"""
YOLACT:Real-time Instance Segmentation
Ref: https://arxiv.org/abs/1904.02689

Arthor: HSU, CHIHCHAO
"""
import tensorflow as tf

from layers.fpn import FeaturePyramidNeck
from layers.head import PredictionModule
from layers.protonet import ProtoNet
from layers.detection import Detect

from data.anchor import Anchor
from config import backbones_objects, backbones_extracted

assert tf.__version__.startswith('2')


class Yolact(tf.keras.Model):
    """
        Creating the YOLCAT Architecture
        Arguments:

    """

    def __init__(self,
                 input_size,
                 backbone,
                 fpn_channels,
                 num_class,
                 num_mask,
                 anchor_params,
                 detect_params):

        super(Yolact, self).__init__()
        # choose the backbone network
        try:
            out = backbones_extracted[backbone]
            base_model = backbones_objects[backbone]
        except:
            raise Exception(f'Backbone option of {backbone} is not supported yet!!!')

        # extract certain feature maps for FPN
        self.backbone_resnet = tf.keras.Model(inputs=base_model.input,
                                              outputs=[base_model.get_layer(x).output for x in out])
        # create remain parts of model
        self.backbone_fpn = FeaturePyramidNeck(fpn_channels)
        self.protonet = ProtoNet(num_mask)

        # semantic segmentation branch to boost feature richness
        # predict num_class - 1
        self.semantic_segmentation = tf.keras.layers.Conv2D(num_class - 1, (1, 1), 1, padding="same",
                                                            kernel_initializer=tf.keras.initializers.glorot_uniform())

        # instance of anchor object
        self.anchor_instance = Anchor(**anchor_params)
        priors = self.anchor_instance.get_anchors()
        # print("prior shape:", priors.shape)
        # print("num anchor per feature map: ", tf.shape(priors)[0])

        # shared prediction head
        self.predictionHead = PredictionModule(256, len(anchor_params["aspect_ratio"]), num_class, num_mask)

        # detection layer
        self.detect = Detect(anchors=priors, **detect_params)

    # Todo need to clarified
    def set_bn(self, mode='train'):
        if mode == 'train':
            for layer in self.backbone_resnet.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
        else:
            for layer in self.backbone_resnet.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

    def call(self, inputs):
        # backbone(ResNet + FPN)
        c3, c4, c5 = self.backbone_resnet(inputs)
        # print("c3: ", c3.shape)
        # print("c4: ", c4.shape)
        # print("c5: ", c5.shape)
        fpn_out = self.backbone_fpn(c3, c4, c5)

        # Protonet branch
        p3 = fpn_out[0]
        protonet_out = self.protonet(p3)
        # print("protonet: ", protonet_out.shape)

        # semantic segmentation branch
        seg = self.semantic_segmentation(p3)

        # Prediction Head branch
        pred_cls = []
        pred_offset = []
        pred_mask_coef = []

        # all output from FPN use same prediction head
        for f_map in fpn_out:
            cls, offset, coef = self.predictionHead(f_map)
            pred_cls.append(cls)
            pred_offset.append(offset)
            pred_mask_coef.append(coef)

        pred_cls = tf.concat(pred_cls, axis=1)
        pred_offset = tf.concat(pred_offset, axis=1)
        pred_mask_coef = tf.concat(pred_mask_coef, axis=1)

        pred = {
            'pred_cls': pred_cls,
            'pred_offset': pred_offset,
            'pred_mask_coef': pred_mask_coef,
            'proto_out': protonet_out,
            'seg': seg
        }

        return pred
