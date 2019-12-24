from models import resnet
import tensorflow as tf
"""
model = resnet.ResNetBackbone101("channel_last")
model.build(input_shape=(None, 224, 224, 3))
model.summary()
"""
base_model = tf.keras.applications.ResNet101(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.summary()
"""