import tensorflow as tf


class FeaturePyramidNeck(tf.keras.layers.Layer):
    """
        Creating the backbone component of feature Pyramid Network
        Arguments:
            num_fpn_filters
    """

    def __init__(self, num_fpn_filters):
        super(FeaturePyramidNeck, self).__init__()
        self.upSample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.downSample1 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="same", activation="relu")
        self.downSample2 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="same", activation="relu")

        self.lateralCov1 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same", activation="relu")
        self.lateralCov2 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same", activation="relu")
        self.lateralCov3 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same", activation="relu")

    def call(self, c3, c4, c5):
        p5 = self.lateralCov1(c5)
        # print("p5: ", p5.shape)
        p6 = self.downSample1(p5)
        # print("p6: ", p6.shape)
        p7 = self.downSample2(p6)
        # print("p7: ", p7.shape)
        p4 = self._crop_and_concat(self.upSample(p5), self.lateralCov2(c4))
        # print("p4: ", p4.shape)
        p3 = self._crop_and_concat(self.upSample(p4), self.lateralCov3(c3))
        # print("p3: ", p3.shape)
        return [p3, p4, p5, p6, p7]

    @staticmethod
    def _crop_and_concat(x1, x2):
        """
        for p4, c4; p3, c3 to concatenate with matched shape
        https://tf-unet.readthedocs.io/en/latest/_modules/tf_unet/layers.html
        """
        x1_shape = x1.shape
        x2_shape = x2.shape
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.add(x1_crop, x2)
