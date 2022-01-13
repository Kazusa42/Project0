import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from yolo_net import YoloBody
from utils import get_classes, cvtColor, resize_image, preprocess_input

COCO_CLASS = r'/coco.names'


class YOLO(object):
    def __init__(self, ):
        self.cls_num = get_classes(COCO_CLASS)
        self.net = YoloBody(self.cls_num)
        self.cuda = True
        self.model_path = r''


    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (416, 416))
        image_data = np.expand_dims(np.transpose(np.array(image_data,
                                                          dtype='float32')), (2, 0, 1), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #  4410 -> 7 * 7 * (10 + 80)
            outputs = self.net(images)
            outputs.reshape((7, 7, 90))
