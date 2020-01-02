import tensorflow as tf
from tensorflow.keras import layers

assert tf.__version__.startswith('2')


class Bottleneck(tf.keras.Model):
    """
        Creating building block for ResNet-50, 101
        Argument:
    """

    def __init__(self, num_filters, data_format, stride, trainable_conv=False, trainable_bn=True):
        super(Bottleneck, self).__init__()
        # Todo Check dataformat (channel first or last)
        axis = -1 if data_format == "channels_last" else 1
        self.trainable_conv = trainable_conv
        self.trainable_bn = trainable_bn

        self.identity = self._conv1x1(num_filters * 4, stride)
        self.bn0 = tf.keras.layers.BatchNormalization(axis=axis)
        self.conv1 = self._conv1x1(num_filters, stride)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=axis)
        self.conv2 = self._conv3x3(num_filters)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=axis)
        self.conv3 = self._conv1x1(num_filters * 4)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=axis)

    def _conv1x1(self, num_filters, stride=1):
        """1 x 1 convolution with padding"""
        return tf.keras.layers.Conv2D(num_filters, (1, 1), stride, padding="same")

    def _conv3x3(self, num_filters, stride=1):
        """3 x 3 convolution with padding"""
        return tf.keras.layers.Conv2D(num_filters, (3, 3), stride, padding="same")

    def call(self, inputs):
        identity = self.identity(inputs)
        identity = self.bn0(identity)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = tf.nn.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = tf.nn.relu(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)

        output = tf.nn.relu(tf.keras.layers.add([identity, bn3]))

        return output


def build_residual_block(num_filters, data_format, num_blocks, stride):
    residual_block = tf.keras.Sequential()
    residual_block.add(Bottleneck(num_filters, data_format, stride))
    for _ in range(1, num_blocks):
        residual_block.add(Bottleneck(num_filters, data_format, stride=1))
    return residual_block


class ResNetBackbone101(tf.keras.Model):
    """
        Creating the backbone component of ResNet-101
        Arguments:

    """

    def __init__(self, data_format):
        super(ResNetBackbone101, self).__init__()
        axis = -1 if data_format == "channels_last" else 1
        self.C1 = tf.keras.Sequential()
        self.C1.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same'))
        self.C1.add(tf.keras.layers.BatchNormalization(axis))
        self.C1.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.C1.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same'))

        self.C2 = build_residual_block(num_filters=64, data_format=data_format, num_blocks=3, stride=1)
        self.C3 = build_residual_block(num_filters=128, data_format=data_format, num_blocks=4, stride=2)
        self.C4 = build_residual_block(num_filters=256, data_format=data_format, num_blocks=23, stride=2)
        self.C5 = build_residual_block(num_filters=512, data_format=data_format, num_blocks=3, stride=2)

    def call(self, inputs):
        C1 = self.C1(inputs)
        print(C1)
        C2 = self.C2(C1)
        print(C2)
        C3 = self.C3(C2)
        print(C3)
        C4 = self.C4(C3)
        print(C4)
        C5 = self.C5(C4)
        print(C5)
        return C3, C4, C5


class ResNetBackbone50(tf.keras.Model):
    """
        Creating the backbone component of ResNet-50
        Arguments:

    """

    def __init__(self):
        # TODO ResNet sub-component
        pass

    def call(self, inputs):
        pass
