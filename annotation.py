import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes
from configure import *

"""
0: the whole process
   including get txt file in dataset/ImageSets and train.txt、val.txt for training
   
1: only get txt file in dataset/ImageSets

2: only get train.txt、val.txt for training
"""
annotation_mode = 0

""" set to the same value when training and predict"""
classes_path = CLASSES_PATH

"""
trainval_percent: the rate of (training set + validation set) : test set
train_percent: the rate of training set : validation set
"""
trainval_percent = 0.9
train_percent = 0.9

dataset_path = DATASET_PATH

VOCdevkit_sets = ['train', 'val']
classes, _ = get_classes(classes_path)


def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(dataset_path, 'Annotations/%s.xml' % image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


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
        for image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(dataset_path, 'ImageSets/Main/%s.txt' % image_set),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s.txt' % image_set, 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/JPEGImages/%s.jpg' % (os.path.abspath(dataset_path), image_id))
                convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate train.txt and val.txt for train done.")
