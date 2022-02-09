# Project0
Codes for Thesis

### Ver 0.1  update on 2022/02/09
1. Complete the code for Faster R-CNN (paper: https://arxiv.org/pdf/1506.01497v3.pdf)
2. predict.py had been tested. train.py has not been tested yet.
3. Creat the first version of README.


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
1. First follow the train part (#Training) to get your own weight.
2. Then change the "_defaults_" param in "frcnn.py".
   _In particulary, "model_path" and "classes_path" these 2 attributes should corresponse to your dataset._
3. Then run predict.py and input the image path.
4. _Other settings about predict is wirtten in predict.py._
