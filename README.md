# Project0
Codes for Thesis

## Versions
### Ver 0.1  update on 2022/02/10
1. Complete the code for Faster R-CNN (paper: https://arxiv.org/pdf/1506.01497v3.pdf)
2. predict.py had been tested, as well as train.py.
3. Creat the first version of README.

### Ver 0.2  update on 2022/03/09
1. get_map.py has been tested and get 73.87% mAP on VOC dataset.
2. simplify some code

## Code structure
1. folder "net" contain the whole network structure of Faster R-CNN (net/frcnn.py).
2. folder "utils" including some support block.
3. class "FRCNN" in frcnn.py is for detection. Its' net comes from "net/frcnn.py".


## Training
__This code use the same format as VOC dataset to train.__
1. Prepare your dataset.  
   Put the images under dir "./VOCdevkit/VOC2007/JPEGImages"  
   Put labels under dir "./VOCdevkit/VOC2007/Annotation"
2. Creat a your_classes.txt file for your dataset and put it under dir "./model_data".
3. Change "classes_path" param in voc_annotation.py, let it corresponse to "./model_data/your_classes.txt"
4. Run voc_annotation.py
5. Change "classes_path" parma in train.py and run train.py.
6. _More details about training is written in train.py_

## Detection
### 1. use pretrained weight
1. Put weight file (.pth) under dir "./model_data". Then run predict.py and input the image path.
### 2. use your own weight
1. First follow the [Training](#Training) part to get your own weight.
2. Then change _"model_path" and "classes_path"_ these 2 attributes of "_defaults_" param in "frcnn.py", let them corresponce to your dataset.
3. Then run predict.py and input the image path.
4. _Other settings about predict is wirtten in predict.py._
