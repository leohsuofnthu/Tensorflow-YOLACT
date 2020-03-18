"""
YOLACT:Real-time Instance Segmentation
Ref: https://arxiv.org/abs/1904.02689

Arthor: HSU, CHIHCHAO
"""
import tensorflow as tf

from layers.fpn import FeaturePyramidNeck
from layers.head import PredictionModule
from layers.protonet import ProtoNet
from utils.create_prior import make_priors

assert tf.__version__.startswith('2')


class Yolact(tf.keras.Model):
    """
        Creating the YOLCAT Architecture
        Arguments:

    """

    def __init__(self, input_size, fpn_channels, feature_map_size, num_class, num_mask, aspect_ratio, scales):
        super(Yolact, self).__init__()
        out = ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        # use pre-trained ResNet50
        # Todo figure out how pre-trained can be train again
        base_model = tf.keras.applications.ResNet50(input_shape=(550, 550, 3),
                                                    include_top=False,
                                                    layers=tf.keras.layers,
                                                    weights='imagenet')
        # extract certain feature maps for FPN
        self.backbone_resnet = tf.keras.Model(inputs=base_model.input,
                                              outputs=[base_model.get_layer(x).output for x in out])
        self.backbone_fpn = FeaturePyramidNeck(fpn_channels)
        self.protonet = ProtoNet(num_mask)

        # semantic segmentation branch to boost feature richness
        self.semantic_segmentation = tf.keras.layers.Conv2D(num_class, (1, 1), 1, padding="same",
                                                            kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.num_anchor, self.priors = make_priors(input_size, feature_map_size, aspect_ratio, scales)
        print("prior shape:", self.priors.shape)
        print("num anchor per feature map: ", self.num_anchor)

        # shared prediction head
        self.predictionHead = PredictionModule(256, len(aspect_ratio), num_class, num_mask)

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
