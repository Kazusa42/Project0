import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes
from configure import *

"""
Params descriptions

annotation_mode:
    0: the whole process
       including get txt file in dataset/ImageSets and train.txt、val.txt for training
    1: only get txt file in dataset/ImageSets
    2: only get train.txt、val.txt for training

trainval_percent:
    the rate of (training set + validation set) : test set

train_percent:
    the rate of training set : validation set

classes_path and dataset_path:
    Set to the same path when trainging.
    classes should BE PAIRED with dataset

ignored_difficulty:
    some objects in the dataset is labeled as "difficult to detected".
    1. True: code will register all objects' information.
    2. False: code will only register those "non-difficult" objects' information. 
"""

annotation_mode = 0
trainval_percent = 1.0
train_percent = 0.9
classes_path = CLASSES_PATH
dataset_path = DATASET_PATH
divided_type = ['train', 'val']
classes, _ = get_classes(classes_path)
ignored_difficulty = True


def convert_annotation(image_name, list_files):
    in_file = open(os.path.join(dataset_path, 'Annotations/%s.xml' % image_name), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if not ignored_difficulty:
            if cls not in classes or int(difficult) == 1:
                continue

        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_files.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(dataset_path, 'Annotations')
        saveBasePath = os.path.join(dataset_path, 'ImageSets/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(range(num), tv)
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in range(num):
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate train.txt and val.txt for train.")
        for image_set in divided_type:
            image_ids = open(os.path.join(dataset_path, 'ImageSets/Main/%s.txt' % image_set),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s.txt' % image_set, 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/JPEGImages/%s.jpg' % (os.path.abspath(dataset_path), image_id))
                convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate train.txt and val.txt for train done.")
