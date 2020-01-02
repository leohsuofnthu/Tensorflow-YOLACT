from models.yolact import Yolact

model = Yolact(256, 256)
model.build(input_shape=(1, 550, 550, 3))
