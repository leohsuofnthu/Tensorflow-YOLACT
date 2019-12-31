"""
YOLACT:Real-time Instance Segmentation
Ref: https://arxiv.org/abs/1904.02689

Arthor: HSU, CHIHCHAO
"""
from models import resnet, fpn, protonet, head
import tensorflow as tf

assert tf.__version__.startswith('2')


class Yolact(tf.keras.Model):
    """
        Creating the YOLCAT Architecture
        Arguments:

    """

    def __init__(self, ):
        self.backbone_resnet = resnet()
        self.backbone_fpn = fpn()
        self.protonet = protonet()
        #Todo Prediction head modules
        pass

    def call(self, inputs):
        pass
