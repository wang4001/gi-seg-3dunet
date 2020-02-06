import argparse
import yaml
import os

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_train_config():
    parser = argparse.ArgumentParser(description='Unet3d training')
    #parser.add_argument('--config', type=str, help='path to the YAML config file')
    parser.add_argument('--config', type=str, default=os.path.expanduser('./configs/train_config_v2.yaml'), 
            help='path to the YAML config file')
    parser.add_argument('--checkpoint_dir', type=str, help='path to save/load checkpoint')
    parser.add_argument('--in_channels', type=int, help='number of input channels')
    parser.add_argument('--out_channels', type=int, help='number of output channels')
    parser.add_argument('--init_channels', type=int, default=32, help='number of feature channels '
                                                                      'in the initial convolution layer')
    parser.add_argument('--interpolate', default=None, help='way to upsample the feature maps')
    parser.add_argument('--loss', type=str, default='dice', help='type of loss function: '
                                                                 'dice-dice, gdc-generalized dice')
    parser.add_argument('--loss_weight', type=float, default=None, nargs='+', help='manual rescaling weights '
                                                                                   'for different classes, '
                                                                                   'e.g. 0.3 0.3 0.4')
    parser.add_argument('--epochs', type=int, default=500, help='max number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='number of validation rounds with no improvement, '
                                                                 'after which the training will be stopped')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')
    parser.add_argument('--validate_after_iters', type=int, default=100, help='number of iterations between validation')
    parser.add_argument('--data_path', type=str, help='path to training dataset')
    parser.add_argument('--train_sub_index', type=str, nargs='+', help='subject index for training in the dataset')
    parser.add_argument('--val_sub_index', type=str, nargs='+', help='subject index for validation in the dataset')
    parser.add_argument('--gpu_index', type=int, nargs='+', help='choose the index of GPU to train the model on')
    parser.add_argument('--train_batch_size', type=int, default=1, help='choose the batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=2, help='choose the batch size for validation')
    parser.add_argument('--norm_type', default=None, help='choose the normalization type')
    parser.add_argument('--concatenate', default=True, help='flag to choose concatenation or addition')
    parser.add_argument('--save_path', type=str, help='path to store results')
    args = parser.parse_args()

    if args.config is not None:
        print('args.config is not None')
        return _load_config_yaml(args.config)
    
    return args


def parse_test_config():
    parser = argparse.ArgumentParser(description='Unet3d testing')
    parser.add_argument('--config', type=str, help='path to the YAML config file')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--save_path', type=str, help='path to save predicted results')
    parser.add_argument('--in_channels', type=int, help='number of input channels')
    parser.add_argument('--out_channels', type=int, help='number of output channels')
    parser.add_argument('--init_channels', type=int, default=32, help='number of feature channels '
                                                                      'in the initial convolution layer')
    parser.add_argument('--interpolate', default=None, help='way to upsample the feature maps')
    parser.add_argument('--data_path', type=str, help='path to testing dataset')
    parser.add_argument('--test_sub_index', type=str, nargs='+', help='subject index for testing in the dataset')
    parser.add_argument('--gpu_index', type=int, nargs='+', help='choose the index of GPU to test the model on')
    parser.add_argument('--test_batch_size', type=int, default=1, help='choose the batch size for testing')
    parser.add_argument('--norm_type', default=None, help='choose the normalization type')
    parser.add_argument('--concatenate', default=True, help='flag to choose concatenation or addition')
    parser.add_argument('--phase', type=str, default='test', help='flag to choose concatenation or addition')

    args = parser.parse_args()

    if args.config is not None:
        config = _load_config_yaml(args.config)
        if args.model_path is not None: 
            config.model_path = args.model_path
        if args.save_path is not None:
            config.save_path = args.save_path
        if args.data_path is not None:
            config.data_path = args.data_path
        if args.test_sub_index is not None:
            config.test_sub_index = args.test_sub_index
        return config

    return args


def _load_config_yaml(config_file):
    config_dict = yaml.load(open(config_file, 'r'), Loader=yaml.SafeLoader)
    return Config(**config_dict)
