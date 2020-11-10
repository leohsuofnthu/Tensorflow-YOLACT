import tensorflow as tf
from data.coco_dataset import prepare_dataloader
from loss.loss_yolact import YOLACTLoss
from yolact import Yolact

# set manual seed for easy debug
# -----------------------------------------------------------------------------------------------
# tf.random.set_seed(1235)

train_dataloader = prepare_dataloader("../data/coco", 12, "train")
loss_fn = YOLACTLoss()

model = Yolact(input_size=550,
               fpn_channels=256,
               feature_map_size=[69, 35, 18, 9, 5],
               num_class=91,
               num_mask=32,
               aspect_ratio=[1, 0.5, 2],
               scales=[24, 48, 96, 192, 384])

# test training sample that contains crowd label
for image, labels in train_dataloader:
    output = model(image, training=True)
    loc_loss, conf_loss, mask_loss, seg_loss, total_loss = loss_fn(output, labels, 91)
    tf.print(f"loc Loss: {loc_loss}, conf Loss: {conf_loss}, mask Loss: {mask_loss}, seg loss: {seg_loss}")
    break
