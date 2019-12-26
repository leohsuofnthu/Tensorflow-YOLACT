import tensorflow as tf
assert tf.__version__.startswith('2')


class FeaturePyramidNeck(tf.keras.Model):
    """
        Creating the backbone component of feature Pyramid Network
        Arguments:
            num_fpn_filters
    """

    def __init__(self, num_fpn_filters):
        super(FeaturePyramidNeck, self).__init__()
        self.upSample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.downSample = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="same")
        self.lateralCov = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same")

    def call(self, c3, c4, c5):
        p5 = self.lateralCov(c5)
        p6 = self.downSample(p5)
        p7 = self.downSample(p6)
        p4 = tf.add(self.upSample(p5), self.lateralCov(c4))
        p3 = tf.add(self.upSample(p4), self.lateralCov(c3))
        return [p3, p4, p5, p6, p7]
