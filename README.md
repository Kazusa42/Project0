# Project0

An implementation of Faster R-CNN.

In order to test the performance get by `AttentionNeck`

More detailes about `AttentionNeck` please refer to [Project1](https://github.com/Kazusa42/Project1)

---

## Usage
The label format is using VOC-style. So, first transform the annotations into `xml` file.

To set the training scheduel, go to `configure.py`, almos every training paramaters are in this script.

Run `train.py` to train the model. If train a model from very begining (do not pre-trained backbone or pre-trained model), set the `epoch` to some big value, such as 500. If use pre-trained backbone or model, `epoch` can be set to some small value to save the time, such as 300.

To evaluate the model, run `get_map.py`. The evaluateing params (such like mAP threshold) are independent from `configure.py`, please directly set them in `get_map.py`.
