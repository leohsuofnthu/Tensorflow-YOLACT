import tensorflow as tf

assert tf.__version__.startswith('2')


class PredictionModule(tf.keras.Model):
    def __init__(self):
        pass

    def call(self, inputs):
        pass
