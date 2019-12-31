from models import resnet
import tensorflow as tf
import numpy as np
"""
model = resnet.ResNetBackbone101("channel_last")
model.build(input_shape=(None, 224, 224, 3))
model.summary()
"""
base_model = tf.keras.applications.ResNet101(input_shape=(550, 550, 3),
                                               include_top=False,
                                               weights='imagenet')

assert tf.test.is_built_with_cuda()
assert tf.test.is_gpu_available()


base_model.summary()
"""
c = ['conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out']

model = tf.keras.Model(inputs=base_model.input, outputs=[base_model.get_layer(x).output for x in c])

image = np.ones((1, 224, 224, 3)).astype(float)

print(model(image))
"""
"""
C3: conv3_block4_out
C4: conv4_block23_out
C5: conv5_block3_out
"""