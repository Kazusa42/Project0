"""
Global variable configure
Aiming to simplify the trainging, predicting and evaluating process between different dataset and weights
"""

BACKBONE_LIST = [r'resnet50',  # 0
                 r'convnext_tiny']  # 1

"""  Basic configure for model """
CLASSES_PATH = r'model_data/DOTA_classes.txt'
DATASET_PATH = r'./DOTA'
MODEL_PATH = r''
BACKBONE = BACKBONE_LIST[1]
IF_CUDA = r'True'
ANCHOR_SIZE = [4, 16, 32]
CONFIDENCE = 0.5
NMS_SCORE = 0.3
INPUT_SHAPE = [640, 640]
RESOLUTION = [INPUT_SHAPE[0] // 32, INPUT_SHAPE[1] // 32]

IF_ATTENTIONNECK = True

""" Training setting """
PRE_TRAINED = True  # If MODEL_PATH is not None, the value will not work.
MOSAIC = True

INIT_EPOCH = 0
FREEZE_EPOCH, UNFREZZEZ_EPOCH = 50, 300
FREEZE_BATCH_SIZE, UNFREEZE_BATCH_SIZE = 16, 6
FREEZE_TRAIN = True

OPT_TYPE = r'adam'  # sgd or adam
INIT_LR = 1e-4  # If you use adam, set this value to 0.001
MIN_LR = INIT_LR * 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0
LR_DECAY_TYPE = r'cos'


""" Others """
FONT_TYPE = r'model_data/font_style_1.ttf'

