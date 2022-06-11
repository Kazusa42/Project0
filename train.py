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
from configure import *


if __name__ == "__main__":
    Cuda = True
    classes_path = CLASSES_PATH

    """
    model_path: load pre-train model.
    
    1. if the train process is interupted, try load params from folder "./logs"
       (remember to change "Freeze_Epoch" and "UnFreeze_Epoch" these 2 params)
       
    2. normally, we do not train a network from very begining
    """
    model_path = MODEL_PATH

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
    anchors_size = ANCHOR_SIZE

    """
    train schedule; Default has 2 stages: frezze and unfrezze
    
    Stage frezze: freeze backbone, only do fine-tune
    Stage unfrezze: update the whole network, including backbone
    """
    init_epoch = 0
    freeze_epoch, unfreeze_epoch = 50, 100
    freeze_batch_size, unfreeze_batch_size = 8, 4
    freeze_lr, unfreeze_lr = 1e-4, 1e-5

    # use stage freeze or not
    freeze_train = True
    
    # thread number
    num_workers = 8

    train_annotation_path = 'train.txt'
    val_annotation_path = 'val.txt'

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
    if freeze_train:
        print('\nStart freeze training.')
        start_epoch = init_epoch
        end_epoch = freeze_epoch
                        
        epoch_step = num_train // freeze_batch_size
        epoch_step_val = num_val // freeze_batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small to train.")
        
        optimizer = optim.Adam(model_train.parameters(), freeze_lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=freeze_batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=freeze_batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
        
        """ freeze backbone """
        if freeze_train:
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
        print('\nStart Un-freeze training.')
        start_epoch = freeze_epoch
        end_epoch = unfreeze_epoch
                        
        epoch_step = num_train // unfreeze_batch_size
        epoch_step_val = num_val // unfreeze_batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small to train.")

        optimizer = optim.Adam(model_train.parameters(), unfreeze_lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=unfreeze_batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=unfreeze_batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
        
        """ Unfreeze those freezed params """
        if freeze_train:
            for param in model.extractor.parameters():
                param.requires_grad = True

        model.freeze_bn()
        train_util = FasterRCNNTrainer(model, optimizer)

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
