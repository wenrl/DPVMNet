import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from trainer import trainer_synapse
from lib.utils_TransUnet import test_single_volume_inference
from tqdm import tqdm
from utils.dataset_synapse import Synapse_dataset, RandomGenerator

from network.DPMamba import DPVMNet 



# from utils import test_single_volume
parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='./data/synapse/preprocessed_synapse_CASCADE/preprocessed_synapse_CASCADE/train_npz_new', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data/synapse/preprocessed_synapse_CASCADE/preprocessed_synapse_CASCADE/test_vol_h5_new', help='root dir for validation volume data')
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

def inference(args, model, test_save_path=None):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print("{} test iterations per epoch".format(len(testloader)))
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model = model.to('cuda')
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        image, label = image.to('cuda'), label.to('cuda')
        metric_i = test_single_volume_inference(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        print('idx %d case %s mean_dice %f mean_hd95 %f mean_iou %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        print('Mean class %d mean_dice %f mean_hd95 %f mean_iou %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_iou : %f' % (performance, mean_hd95, mean_iou))
    return "Testing Finished!"

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

    model = DPVMNet(in_channels=1, out_channels=9) 

    model.load_state_dict(torch.load('./best.pth')) #best.pth
    print('Model successfully created.')
    test_save_path = './outputs/synapse/ours'
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    inference(args, model, test_save_path)
