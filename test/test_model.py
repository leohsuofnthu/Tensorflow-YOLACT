from config import get_params
from yolact import Yolact
import tensorflow as tf

name = 'coco'
train_iter, input_size, num_cls, lrs_schedule_params, loss_params, parser_params, model_params = get_params(
    name)
model = Yolact(**model_params)
_ = model(tf.zeros([2, 550, 550, 3]), training=False)
model.summary()
