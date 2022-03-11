"""
Global variable configure

Aiming to simplify the trainging, predicting and evaluating process between different dataset
"""

# frcnn basic configure
CLASSES_PATH = r'model_data/visdrone_classes.txt'
MODEL_PATH = r'model_data/visdrone_init.pth'
ANCHOR_SIZE = [4, 16, 32]  # [4, 16, 32] for tiny object, [8, 16, 32] for normal type
IF_CUDA = r'True'
FONT_TYPE = r'model_data/monoMMM_5.ttf'
#  get_map
pass

#  Dataset
DATASET_PATH = r'VisDrone'
