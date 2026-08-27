# coding=utf-8
import argparse
import random
import os
from model import WiFo2_model, WiFo2
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
        task_id = 3,
        note = '',
        data_path = './dataset',
        wifo_weights = './weights/model_best.pkl',
        fastdepth_weights = './weights/FastDepthV2_L1_Best.pth',
        process_name = 'process_name',
        snr_db = 25,
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
        epochs = 200,
        early_stop = 5,
        weight_decay = 0,
        batch_size = 64,
        log_interval = 5,
        device_id = '0',
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

    train_data, test_data = data_load(args)

    args.folder = '{}_Task{}_{}_{}/'.format(args.task, args.task_id, args.size, args.note)
    args.model_path = './experiments/{}'.format(args.folder)
    logdir = "./logs/{}".format(args.folder)
    os.makedirs(args.model_path + 'model_save/', exist_ok=True)

    writer = SummaryWriter(log_dir=logdir, flush_secs=5)
    device = dev(args.device_id)
    model = WiFo2_model(args=args).to(device)

    msg = model.load_state_dict(torch.load(args.wifo_weights, map_location=device), strict=False)
    # print(msg)
    model = freeze_wifo(args, model, args.task_id)

    # 优化器 + lr调度器
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], args.lr)
    scheduler = FakeLR(optimizer=optimizer)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params / 1e6} M')
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of learnable parameter: {total_learn / 1e6} M')

    TrainLoop(
        args=args,
        writer=writer,
        model=model,
        optimizer=optimizer, 
        scheduler=scheduler,
        train_data=train_data,
        test_data=test_data,
        device=device,
        early_stop=args.early_stop,
    ).run()


def freeze_wifo(args, model: WiFo2, task_id):    
    for param in model.parameters():
        param.requires_grad = False
    if task_id == 1:
        for param in model.Fine_Tune_Layer_LoS_NLoS.parameters():
            param.requires_grad = True
    if task_id == 2:
        for param in model.res18_align_layer.parameters():
            param.requires_grad = True
        for param in model.pos_align_layer.parameters():
            param.requires_grad = True
        for param in model.decoder_blocks[0].parameters():
            param.requires_grad = True
        for param in model.decoder_pred.parameters():
            param.requires_grad = True
    if task_id == 3:
        for layer_id in range(14, 18):
            for param in getattr(model.fast_depth.encoder, f'enc_layer{layer_id}').parameters():
                param.requires_grad = True
        for param in model.fast_depth.decoder.parameters():
            param.requires_grad = True
        if args.csi_enhance:
            for param in model.Fine_Tune_Layer0_DE.parameters():
                param.requires_grad = True
            for param in model.Fine_Tune_Layer1_DE.parameters():
                param.requires_grad = True
    
    return model


if __name__ == "__main__":
    main()
