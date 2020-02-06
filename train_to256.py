import torch
import torch.nn as nn
import torch.optim

from model_v4 import Unet3d
from small_model import smallmodel
from loss import DiceAccuracy, DiceLoss
from data import Hdf5Dataset
from torch.utils.data import DataLoader
from config_v1 import parse_train_config
from trainer_v3_trainto256_aug import Trainer
from initialize_weight import init_weight
import utils

torch.cuda.manual_seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    # load the config
    config = parse_train_config()
    # load the model
    model = Unet3d(in_channels=config.in_channels, out_channels=config.out_channels, interpolate=config.interpolate,
                   concatenate=config.concatenate,
                   norm_type=config.norm_type, init_channels=config.init_channels, scale_factor=(2, 2, 2))
   
    if config.init_weight:
        model.apply(init_weight)
    
    # get the device to train on
    gpu_all = tuple(config.gpu_index)
    gpu_main = gpu_all[0]

    device = torch.device('cuda:' + str(gpu_main) if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(model, device_ids=gpu_all)
    model.to(device)
    
    # load the saved checkpoint - update model parameters
    utils.load_checkpoint(config.model_path, device, model)
    for params in model.parameters():
        params.requires_grad = False

    model2 = smallmodel(out_channels=config.out_channels, interpolate=config.interpolate, norm_type=config.norm_type, init_channels=config.init_channels)
    
    if config.init_weight:
        model2.apply(init_weight)
    
    model2 = nn.DataParallel(model2, device_ids=gpu_all)
    model2.to(device)

    # load data
    phase = 'train'
    train_dataset = Hdf5Dataset(config.data_path, phase, config.train_sub_index)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True,
            num_workers=8, pin_memory=True, drop_last=True)
    val_dataset = Hdf5Dataset(config.data_path, phase, config.val_sub_index)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.val_batch_size, shuffle=False, 
            num_workers=8, pin_memory=True, drop_last=True)

    # define accuracy
    accuracy_criterion = DiceAccuracy()

    # define loss
    if config.loss_weight is None:
        loss_criterion = DiceLoss()
    else:
        loss_criterion = DiceLoss(weight=config.loss_weight)

    # define optimizer
    optimizer = torch.optim.Adam(model2.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    trainer = Trainer(config, model, model2, device, train_loader, val_loader, accuracy_criterion, loss_criterion, optimizer)
    trainer.main()

if __name__ == '__main__':
    main()
