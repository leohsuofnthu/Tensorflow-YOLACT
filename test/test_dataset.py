import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import PASCAL_CLASSES, COCO_CLASSES, COLORS, get_params, ROOT_DIR
from data.coco_dataset import ObjectDetectionDataset
from yolact import Yolact

# Todo Add your custom dataset
NAME_OF_DATASET = "pascal"
CLASS_NAMES = PASCAL_CLASSES
# -----------------------------------------------------------------------------------------------
# create model and dataloader
train_iter, input_size, num_cls, lrs_schedule_params, loss_params, parser_params, model_params = get_params(
    NAME_OF_DATASET)
model = Yolact(**model_params)
dateset = ObjectDetectionDataset(dataset_name=NAME_OF_DATASET,
                                 tfrecord_dir=os.path.join(ROOT_DIR, "data", NAME_OF_DATASET),
                                 anchor_instance=model.anchor_instance,
                                 **parser_params)
train_dataset = dateset.get_dataloader(subset='train', batch_size=1)
valid_dataset = dateset.get_dataloader(subset='val', batch_size=1)

# Visualize one image from training set
for image, labels in train_dataset.take(1):
    ori = labels['ori'].numpy()[0]
    image = image.numpy()[0].astype(np.uint8)
    plt.figure(figsize=(7, 7))
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    boxes = labels['bbox'].numpy()[0]
    classes = labels['classes'].numpy()[0]
    masks = labels['mask_target'].numpy()[0]
    final_m = np.zeros_like(masks[0][:, :, None])
    for box, _cls, mask in zip(boxes, classes, masks):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w == 0 or h == 0:
            break

        class_id = _cls - 1
        color_idx = (class_id * 5) % len(COLORS)
        color = [c / 255.0 for c in COLORS[color_idx]]
        text = f"{CLASS_NAMES[class_id]}"

        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=1
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.6},
            clip_box=ax.clipbox,
            clip_on=True,
        )
        m = mask[:, :, None]
        final_m = final_m + np.concatenate((m * color[0], m * color[1], m * color[2]), axis=-1)

    plt.figure()
    plt.imshow(ori)

    plt.figure()
    plt.imshow(np.asarray(final_m))
    plt.show()
