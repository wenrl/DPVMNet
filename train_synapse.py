import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from network.DPMamba import DPVMNet

from trainer import trainer_synapse, trainer_wrl_synapse

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='./synapse/preprocessed_synapse_CASCADE/preprocessed_synapse_CASCADE/train_npz_new', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./synapse/preprocessed_synapse_CASCADE/preprocessed_synapse_CASCADE/test_vol_h5_new', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')

parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=6, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    
    if args.concatenation:
        aggregation = 'concat'
    else: 
        aggregation = 'add'
    
    if args.no_dw_parallel:
        dw_mode = 'series'
    else: 
        dw_mode = 'parallel'
    
    run = 1

    
    snapshot_path = './model_pth'

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    model = DPVMNet(in_channels=1, out_channels=9)
    model.cuda()

    print('Model successfully created.')
    
    # trainer = {'Synapse': trainer_synapse,}
    trainer = {'Synapse': trainer_wrl_synapse,}
    trainer[dataset_name](args, model, snapshot_path)
