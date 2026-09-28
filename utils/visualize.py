"""Helpers for drawing YOLACT detections on images."""
import cv2
import numpy as np

from config import COLORS


def get_class_names(dataset_name):
    from config import COCO_CLASSES, PASCAL_CLASSES, YOUR_CUSTOM_CLASSES

    mapping = {
        'coco': COCO_CLASSES,
        'pascal': PASCAL_CLASSES,
    }
    if dataset_name in mapping:
        return mapping[dataset_name]
    if YOUR_CUSTOM_CLASSES:
        return YOUR_CUSTOM_CLASSES
    return tuple(f'class_{i}' for i in range(200))


def _color(idx):
    return COLORS[int(idx) % len(COLORS)]


def draw_detections(image_bgr,
                    classes,
                    scores,
                    boxes,
                    masks,
                    class_names,
                    score_threshold=0.3,
                    mask_alpha=0.45):
    """
    Draw instance masks, boxes, and labels on a BGR image.

    boxes: [N, 4] in (xmin, ymin, xmax, ymax) of the *same* resolution as image_bgr.
    masks: [N, H, W] binary / soft masks matching image_bgr spatial size.
    """
    out = image_bgr.copy()
    if classes is None:
        return out

    classes = np.asarray(classes).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    boxes = np.asarray(boxes).reshape(-1, 4)
    if masks is not None:
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]

    keep = scores > score_threshold
    classes, scores, boxes = classes[keep], scores[keep], boxes[keep]
    if masks is not None:
        masks = masks[keep]

    order = np.argsort(scores)
    for i in order:
        color = _color(classes[i])
        color_arr = np.array(color, dtype=np.float32)

        if masks is not None and i < len(masks):
            mask = masks[i]
            if mask.shape[:2] != out.shape[:2]:
                mask = cv2.resize(mask.astype(np.float32),
                                  (out.shape[1], out.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            mask_bool = mask > 0.5
            out[mask_bool] = (
                out[mask_bool].astype(np.float32) * (1.0 - mask_alpha) + color_arr * mask_alpha
            ).astype(np.uint8)

        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        cls_idx = int(classes[i])
        name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
        label = f'{name}: {scores[i]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out
