import tensorflow as tf
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import ResizeMethod


class FeaturePyramidNeck(tf.keras.layers.Layer):
    """
        Creating the backbone component of feature Pyramid Network
        Arguments:
            num_fpn_filters
    """

    def __init__(self, num_fpn_filters):
        super(FeaturePyramidNeck, self).__init__()
        self.upSample = tf.keras.layers.UpSampling2D(2)

        # no Relu for downsample layer
        self.downSample1 = tf.keras.layers.Conv2D(num_fpn_filters, 3, 2, "same")
        self.downSample2 = tf.keras.layers.Conv2D(num_fpn_filters, 3, 2, "same")

        self.lateralCov1 = tf.keras.layers.Conv2D(num_fpn_filters, 1, 1, 'valid')
        self.lateralCov2 = tf.keras.layers.Conv2D(num_fpn_filters, 1, 1, 'valid')
        self.lateralCov3 = tf.keras.layers.Conv2D(num_fpn_filters, 1, 1, 'valid')

        # predict layer for FPN
        self.predictP5 = tf.keras.layers.Conv2D(num_fpn_filters, 3, 1, "same")
        self.predictP4 = tf.keras.layers.Conv2D(num_fpn_filters, 3, 1, "same")
        self.predictP3 = tf.keras.layers.Conv2D(num_fpn_filters, 3, 1, "same")

    def call(self, c3, c4, c5):
        # lateral conv for c3 c4 c5
        p5 = self.lateralCov1(c5)
        p4 = self.lateralCov2(c4)
        p3 = self.lateralCov3(c3)

        p4 = self._resize_and_add(p5, p4)
        p3 = self._resize_and_add(p4, p3)

        # smooth pred layer for p3, p4, p5
        p3 = tf.nn.relu(self.predictP3(p3))
        p4 = tf.nn.relu(self.predictP4(p4))
        p5 = tf.nn.relu(self.predictP5(p5))

        # downsample conv to get p6, p7
        p6 = self.downSample1(p5)
        p7 = self.downSample2(tf.nn.relu(p6))

        return [p3, p4, p5, p6, p7]

    def _resize_and_add(self, f1, f2):
        _, h, w, _ = f2.shape
        return tf.image.resize(f1, (h, w), method=ResizeMethod.BILINEAR, preserve_aspect_ratio=False) + f2
