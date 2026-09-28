import os

import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------
# Set the hyperparameter you preferred
MIXPRECISION = False
RANDOM_SEED = 1234

# Parser
NUM_MAX_PAD = 100
THRESHOLD_POS = 0.5
THRESHOLD_NEG = 0.4

# Model
BACKBONE = 'resnet50'
IMG_SIZE = 550
PROTO_OUTPUT_SIZE = 138
FPN_CHANNELS = 256
NUM_MASK = 32

# Loss (paper defaults)
LOSS_WEIGHT_CLS = 1
LOSS_WEIGHT_BOX = 1.5
LOSS_WEIGHT_MASK = 6.125
LOSS_WEIGHT_SEG = 1
NEG_POS_RATIO = 3
MAX_MASKS_FOR_TRAIN = 100

# Detection
TOP_K = 200
CONF_THRESHOLD = 0.05
NMS_THRESHOLD = 0.3
MAX_NUM_DETECTION = 100

# -----------------------------------------------------------------
# Backbone feature map layer names for FPN (approx. strides 8 / 16 / 32)

backbones_extracted = {
    'resnet50': ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'],
    'resnet101': ['conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'],
    'mobilenetv2': ['block_5_add', 'block_7_add', 'block_14_add'],
    'efficientNet-B0': ['block3b_add', 'block4c_add', 'block6d_add'],
}

backbones_preprocess = {
    'resnet50': tf.keras.applications.resnet50.preprocess_input,
    'resnet101': tf.keras.applications.resnet50.preprocess_input,
    'mobilenetv2': tf.keras.applications.mobilenet_v2.preprocess_input,
    'efficientNet-B0': tf.keras.applications.efficientnet.preprocess_input,
}


def build_backbone(name, img_size=IMG_SIZE, weights='imagenet'):
    """Lazily construct a single backbone (avoids loading all models on import)."""
    builders = {
        'resnet50': tf.keras.applications.ResNet50,
        'resnet101': tf.keras.applications.ResNet101,
        'mobilenetv2': tf.keras.applications.MobileNetV2,
        'efficientNet-B0': tf.keras.applications.EfficientNetB0,
    }
    if name not in builders:
        raise ValueError(
            f"Backbone '{name}' is not supported. Choose from: {list(builders)}"
        )
    if name not in backbones_extracted:
        raise ValueError(f"No FPN extraction layers defined for backbone '{name}'.")

    base = builders[name](
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=weights,
    )
    return base, backbones_extracted[name]


# Kept for older tests / notebooks that imported these names.
backbones_objects = None  # use build_backbone() instead

# RGB values for drawing boxes / masks
COLORS = ((244, 67, 54),
          (233, 30, 99),
          (156, 39, 176),
          (103, 58, 183),
          (63, 81, 181),
          (33, 150, 243),
          (3, 169, 244),
          (0, 188, 212),
          (0, 150, 136),
          (76, 175, 80),
          (139, 195, 74),
          (205, 220, 57),
          (255, 235, 59),
          (255, 193, 7),
          (255, 152, 0),
          (255, 87, 34),
          (121, 85, 72),
          (158, 158, 158),
          (96, 125, 139))

# -----------------------------------------------------------------
# Dataset-specific settings
# For a custom dataset: copy the "custom" entries and fill them in.

NUM_CLASSES = {
    'coco': 81,       # 80 classes + background
    'pascal': 21,     # 20 classes + background
    'custom': 0,      # TODO: set to (num_classes + 1) including background
}

TRAIN_ITER = {
    'coco': 800000,
    'pascal': 120000,
    'custom': 100000,  # TODO: adjust for dataset size
}

LR_STAGE = {
    'coco': {
        'warmup_steps': 500,
        'warmup_lr': 1e-4,
        'initial_lr': 1e-3,
        'stages': [280000, 600000, 700000, 750000],
        'stage_lrs': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
    },
    'pascal': {
        'warmup_steps': 500,
        'warmup_lr': 1e-4,
        'initial_lr': 1e-3,
        'stages': [60000, 100000],
        'stage_lrs': [1e-3, 1e-4, 1e-5],
    },
    # Sensible default schedule for small/medium custom datasets
    'custom': {
        'warmup_steps': 500,
        'warmup_lr': 1e-4,
        'initial_lr': 1e-3,
        'stages': [60000, 80000],
        'stage_lrs': [1e-3, 1e-4, 1e-5],
    },
}

ANCHOR = {
    'coco': {
        'img_size': IMG_SIZE,
        'feature_map_size': [69, 35, 18, 9, 5],
        'aspect_ratio': [1, 0.5, 2],
        'scale': [24, 48, 96, 192, 384],
    },
    'pascal': {
        'img_size': IMG_SIZE,
        'feature_map_size': [69, 35, 18, 9, 5],
        'aspect_ratio': [1, 0.5, 2],
        'scale': [24 * (4 / 3), 48 * (4 / 3), 96 * (4 / 3), 192 * (4 / 3), 384 * (4 / 3)],
    },
    'custom': {
        'img_size': IMG_SIZE,
        'feature_map_size': [69, 35, 18, 9, 5],
        'aspect_ratio': [1, 0.5, 2],
        'scale': [24, 48, 96, 192, 384],
    },
}

# Optional: class name tuples used by visualization
YOUR_CUSTOM_CLASSES = ()  # e.g. ('cat', 'dog')

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

PASCAL_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor')

# Map original COCO category ids (1..90) -> contiguous model ids (1..80)
COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

LABEL_MAP = {
    'coco': COCO_LABEL_MAP,
    'pascal': None,
    'custom': None,  # set a dict if your annotation ids are not already 1..N
}


def get_params(dataset_name):
    if dataset_name not in NUM_CLASSES:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Add it in config.py (NUM_CLASSES / TRAIN_ITER / LR_STAGE / ANCHOR / LABEL_MAP)."
        )
    if NUM_CLASSES[dataset_name] < 2:
        raise ValueError(
            f"NUM_CLASSES['{dataset_name}'] must be >= 2 (classes + background). "
            'Edit config.py before training a custom dataset.'
        )

    parser_params = {
        'output_size': IMG_SIZE,
        'proto_out_size': PROTO_OUTPUT_SIZE,
        'num_max_padding': NUM_MAX_PAD,
        'augmentation_params': {
            'preprocess_func': backbones_preprocess[BACKBONE],
            'mean': (0.407, 0.457, 0.485),
            'std': (0.225, 0.224, 0.229),
            'output_size': IMG_SIZE,
            'proto_output_size': PROTO_OUTPUT_SIZE,
            'discard_box_width': 4. / float(IMG_SIZE),
            'discard_box_height': 4. / float(IMG_SIZE),
        },
        'matching_params': {
            'threshold_pos': THRESHOLD_POS,
            'threshold_neg': THRESHOLD_NEG,
        },
        'label_map': LABEL_MAP[dataset_name],
    }

    detection_params = {
        'num_cls': NUM_CLASSES[dataset_name],
        'label_background': 0,
        'top_k': TOP_K,
        'conf_threshold': CONF_THRESHOLD,
        'nms_threshold': NMS_THRESHOLD,
        'max_num_detection': MAX_NUM_DETECTION,
    }

    loss_params = {
        'loss_weight_cls': LOSS_WEIGHT_CLS,
        'loss_weight_box': LOSS_WEIGHT_BOX,
        'loss_weight_mask': LOSS_WEIGHT_MASK,
        'loss_weight_seg': LOSS_WEIGHT_SEG,
        'neg_pos_ratio': NEG_POS_RATIO,
        'max_masks_for_train': MAX_MASKS_FOR_TRAIN,
    }

    model_params = {
        'backbone': BACKBONE,
        'fpn_channels': FPN_CHANNELS,
        'num_class': NUM_CLASSES[dataset_name],
        'num_mask': NUM_MASK,
        'anchor_params': ANCHOR[dataset_name],
        'detect_params': detection_params,
    }

    return (TRAIN_ITER[dataset_name], IMG_SIZE, NUM_CLASSES[dataset_name],
            LR_STAGE[dataset_name], loss_params, parser_params, model_params)
