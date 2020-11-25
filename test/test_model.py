import tensorflow as tf
from yolact import Yolact

IMG_SIZE = 550
model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          include_top=False,
                                          layers=tf.keras.layers,
                                          weights='imagenet')

# model = Yolact(**cfg.model_parmas)
model.build(input_shape=(2, IMG_SIZE, IMG_SIZE, 3))
model.summary()
