# coding=utf-8
import argparse
import random
import os
from model_T1 import WiFo2_model as build_model_T1
from model_T2 import WiFo2_model as build_model_T2
from model_T3 import WiFo2_model as build_model_T3
from train import TrainLoop
import setproctitle
import torch
from DataLoader import data_load
from utils import *
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scheduler import FakeLR


def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device('cuda:{}'.format(device_id))
    return torch.device("cpu")

def create_argparser():
    defaults = dict(
        # experimental settings
        task = 'Train',
        task_id = 2,
        note = '',
        data_path = './dataset',
        wifo_weights = './weights/model_best.pkl',
        fastdepth_weights = './weights/FastDepthV2_L1_Best.pth',
        process_name = 'process_name',
        csi_enhance = 1,
        rgb_pos_enhance = 1,

        # model settings
        MoE = False,
        patch_size = 4,
        t_patch_size = 4,
        size = 'tinypro',
        no_qkv_bias = 0,
        pos_emb = 'SinCos_3D',
        
        # training parameters
        lr = 1e-5,
        min_lr = 1e-6,
        epochs = 1000,
        early_stop = 5,
        weight_decay = 0,
        batch_size = 64,
        log_interval = 5,
        device_id = '2',
        machine = 'machine_name',
        clip_grad = 0.05,
        lr_anneal_steps = 200,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    torch.autograd.set_detect_anomaly(True)
    
    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id))
    setup_init(100)

    device = dev(args.device_id)
    model_T1 = build_model_T1(args=args).to(device)
    model_T2 = build_model_T2(args=args).to(device)
    model_T3 = build_model_T3(args=args).to(device)

    print(f'Task 1 number of parameters: {sum(p.numel() for p in model_T1.parameters()) / 1e6} M')
    print(f'Task 2 number of parameters: {sum(p.numel() for p in model_T2.parameters()) / 1e6} M')
    print(f'Task 3 number of parameters: {sum(p.numel() for p in model_T3.parameters()) / 1e6} M')


if __name__ == "__main__":
    main()
