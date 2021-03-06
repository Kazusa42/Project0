import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from frcnn import FRCNN
from utils.utils import get_classes
from utils.utils_map import get_map
from configure import *

if __name__ == "__main__":
    """
    param: map_mode use to control what we calculate
    0: the whole process, including get predict result, get ground truth and calculate VOC mAP
    1: only get predict result
    2: only get ground truth
    3: only calculate VOC mAP
    """
    map_mode = 0

    """
    params
    classes_path: normally just set it as the same path when training or predict
    MINOVERLAP: iou, the mAP will be calculated under this iou
    map_vis: visualization or not
    map_out_path: mAP output file path 
    """
    classes_path = CLASSES_PATH
    dataset_path = DATASET_PATH
    MINOVERLAP = 0.5
    map_vis = False
    map_out_path = 'map_out'

    image_ids = open(os.path.join(dataset_path, "ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        frcnn = FRCNN(confidence=0.01, nms_iou=0.5)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(dataset_path, "JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            frcnn.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(dataset_path, "Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path=map_out_path)
        print("Get map done.")


