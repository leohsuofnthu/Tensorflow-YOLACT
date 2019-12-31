from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
assert tf.__version__.startswith('2')


class ProtoNet(tf.keras.Model):
    """
        Creating the component of ProtoNet
        Arguments:
            num_prototype
    """

    def __init__(self,  num_prototype):
        super(ProtoNet, self).__init__()
        self.initialConv = tf.keras.layers.Conv2D(256, (3, 3), 2, padding="same")
        self.finalConv = tf.keras.layers.Conv2D(num_prototype, (3, 3), 2, padding="same")
        self.upSampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
    def call(self, p3):
        p3 = self.initialConv(p3)
        p3 = self.initialConv(p3)

        pass
