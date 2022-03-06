import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from net.frcnn import FasterRCNN
from net.frcnn_training import FasterRCNNTrainer, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

'''
训练自己的目标检测模型一定需要注意以下几点：
1. 训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2. 训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3. 损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4. 调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。

5. Good Luck
'''  

if __name__ == "__main__":
    Cuda = True

    # if use your own dataset to train, please change this path.
    classes_path = r'model_data/voc_classes.txt'

    """
    model_path: load pre-train model.
    
    1. if the train process is interupted, try load params from folder "./logs"
       (remember to change "Freeze_Epoch" and "UnFreeze_Epoch" these 2 params)
       
    2. normally, we do not train a network from very begining
    """
    model_path = r'model_data/voc_weights_resnet.pth'

    input_shape = [600, 600]
    backbone = r"resnet50"

    """
    param: pretrained; use or not use pretrained backbone
    
    NOTICE:
    if model_path is not None, pretrained is default as True.
    if model_path is None, for most situations, please set pretrained as True.
    
    model_path will load weights for whole network, pretrained will only load weights for backbone
    """
    pretrained = False

    """ 
    if want to detecte small objects, use small value for the font-number in anchors_size
    e.g. anchors_size = [4, 16, 32]
    """
    anchors_size = [8, 16, 32]

    """
    train schedule; Default has 2 stages: frezze and unfrezze
    
    Stage frezze: freeze backbone, only do fine-tune
    Stage unfrezze: update the whole network, including backbone
    """
    Init_Epoch = 0
    Freeze_Epoch, UnFreeze_Epoch = 50, 100
    Freeze_batch_size, Unfreeze_batch_size = 4, 2
    Freeze_lr, Unfreeze_lr = 1e-4, 1e-5

    # use stage freeze or not
    Freeze_Train = True
    
    # thread number
    num_workers = 4

    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    class_names, num_classes = get_classes(classes_path)
    
    model = FasterRCNN(num_classes, anchor_scales=anchors_size, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory("logs/")

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    """ stage freeze """
    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch
                        
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small to train.")
        
        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
        
        """ freeze backbone """
        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = False

        # freeze bn layer
        model.freeze_bn()

        train_util = FasterRCNNTrainer(model, optimizer)

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
    
    """ stage unfrezze"""
    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch
                        
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small to train.")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
        
        """ Unfreeze those freezed params """
        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = True

        model.freeze_bn()
        train_util = FasterRCNNTrainer(model, optimizer)

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
