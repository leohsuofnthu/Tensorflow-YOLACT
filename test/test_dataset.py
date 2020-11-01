import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data.coco_dataset import prepare_dataloader
from utils.utils import denormalize_image
from utils.label_map import COCO_LABEL_MAP, COCO_CLASSES, COLORS

# set manual seed for easy debug
# -----------------------------------------------------------------------------------------------
tf.random.set_seed(1235)

train_dataloader = prepare_dataloader("../data/coco", 1, "train")
print(train_dataloader)
# visualize the training sample
for image, labels in train_dataloader.take(1):
    image = denormalize_image(image)
    image = np.squeeze(image.numpy()) * 255
    image = image.astype(np.uint8)
    print(image.max(), image.min())
    bbox = labels['bbox'].numpy()
    cls = labels['classes'].numpy()
    mask = labels['mask_target'].numpy()
    num_obj = labels['num_obj'].numpy()
    # original_img = np.squeeze(labels['ori'].numpy().astype(np.uint8))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    final_m = np.zeros_like(mask[0][0][:, :, None])
    for idx in range(num_obj[0]):
        # get the bbox, class_name, and random color
        b = bbox[0][idx]
        m = mask[0][idx][:, :, None]
        class_id = COCO_LABEL_MAP.get(cls[0][idx]) - 1
        color_idx = (class_id * 5) % len(COLORS)

        # prepare the class text to display
        text_str = f"{COCO_CLASSES[class_id]}"
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        text_pt = (int(b[1]), int(b[0] - 3))
        text_color = [255, 255, 255]
        color = COLORS[color_idx]

        # draw the bbox, text, and bbox around text
        cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), color, 1)
        cv2.rectangle(image, (b[1], b[0]), (int(b[1] + text_w), int(b[0] - text_h - 4)), color, -1)
        cv2.putText(image, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # create mask
        final_m = final_m + np.concatenate((m * color[0], m * color[1], m * color[2]), axis=-1)

    final_m = final_m.astype('uint8')
    dst = np.zeros_like(image).astype('uint8')
    final_m = cv2.resize(final_m, dsize=(image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    cv2.addWeighted(final_m, 0.3, image, 0.7, 0, dst)
    cv2.imshow("check", dst)
    k = cv2.waitKey(0)
