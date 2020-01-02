"""
YOLACT:Real-time Instance Segmentation
Ref: https://arxiv.org/abs/1904.02689

Arthor: HSU, CHIHCHAO
"""
from models.fpn import FeaturePyramidNeck
from models.protonet import ProtoNet
from models.head import PredictionModule
import tensorflow as tf

assert tf.__version__.startswith('2')


class Yolact(tf.keras.Model):
    """
        Creating the YOLCAT Architecture
        Arguments:

    """

    def __init__(self, fpn_channels, selected_layer_pred=0, num_class=1, num_mask=4, aspect_ratio=1, scale=1):
        super(Yolact, self).__init__()
        out = ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        # use pre-trained ResNet50
        # Todo figure out how pre-trained can be train again
        base_model = tf.keras.applications.ResNet50(input_shape=(550, 550, 3),
                                                    include_top=False,
                                                    weights='imagenet')
        # extract certain feature maps for FPN
        self.backbone_resnet = tf.keras.Model(inputs=base_model.input,
                                              outputs=[base_model.get_layer(x).output for x in out])
        self.backbone_fpn = FeaturePyramidNeck(fpn_channels)
        self.protonet = ProtoNet(num_mask)
        # Todo Prediction head modules

    def call(self, inputs):
        c3, c4, c5 = self.backbone_resnet(inputs)
        print("c3: ", c3.shape)
        print("c4: ", c4.shape)
        print("c5: ", c5.shape)
        fpn_out = self.backbone_fpn(c3, c4, c5)
        p3 = fpn_out[0]
        protonet_out = self.protonet(p3)
        print("protonet: ", protonet_out.shape)
        return protonet_out
