from config import get_params
from yolact import Yolact

name = "coco"
train_iter, input_size, num_cls, lrs_schedule_params, loss_params, parser_params, model_params = get_params(
    name)
model = Yolact(**model_params)
model.build(input_shape=(2, 550, 550, 3))
model.summary()
