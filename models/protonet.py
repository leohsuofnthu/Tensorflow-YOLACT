from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

assert tf.__version__.startswith('2')


class ProtoNet(tf.keras.Model):
    """
        Creating the component of Protonet
        Arguments:

    """

    def __init__(self):
        super(ProtoNet, self).__init__()
        pass

    def call(self, inputs):
        pass
