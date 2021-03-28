import tensorflow as tf

model = tf.keras.applications.ResNet50(input_shape=(550, 550, 3),
                                       include_top=False,
                                       weights='imagenet')

print(model.summary())