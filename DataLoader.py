# coding=utf-8
import numpy as np
import torch as th
import os
import json
import torch
import scipy.io
import datetime
import copy
import hdf5storage
import random
import math
from torch.utils.data import Dataset
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt


def LoadBatch(H):
    # H: ...     [tensor complex]
    # out: ..., 2  [tensor real]
    size = list(H.shape)
    H_real = np.zeros(size + [2])
    H_real[..., 0] = H.real
    H_real[..., 1] = H.imag
    H_real = torch.tensor(H_real, dtype=torch.float32)
    if len(H_real.shape) == 4:
        H_real = rearrange(H_real, 'b n k o -> b o n k')
    else:
        H_real = rearrange(H_real, 'b t n k o -> b o t n k')
    return H_real


class Dataset_task_1(Dataset):
    def __init__(self, file_path, itr=200, SEED=42, is_show=1, dataset_name="train"):
        super(Dataset_task_1, self).__init__()
        # Shuffle and Segmentation Dateset
        db = hdf5storage.loadmat(file_path + f'/H_{dataset_name}.mat')[f'H']
        db2 = hdf5storage.loadmat(file_path + f'/L_{dataset_name}.mat')[f'label']
        Batch_num = db.shape[0]
        np.random.seed(SEED)
        idx = np.arange(Batch_num)
        np.random.shuffle(idx)
        idx = idx[:itr]

        self.H_full = torch.tensor(db, dtype=torch.complex128)
        self.Label = torch.tensor(db2, dtype=torch.long)

        self.H_full = LoadBatch(self.H_full)

        self.H_full = self.H_full[idx, ...]
        self.Label = self.Label[idx, ...]

        if is_show:
            print('Dataset info: ')
            print(
                f'H_full (input) shape: {self.H_full.shape}\t'
                f'label shape: {self.Label.shape}\n'
            )

    def __getitem__(self, index):
        return {
            "label": self.Label[index].long(),
            "H_full": self.H_full[index, ...].float(),
        }

    def __len__(self):
        return self.H_full.shape[0]


def data_load_task_1(args):
    train_data = Dataset_task_1(args.data_path + '/Task1', itr=10, dataset_name="train")
    test_data = Dataset_task_1(args.data_path + '/Task1', itr=200, dataset_name="test")

    train_data = th.utils.data.DataLoader(train_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                          pin_memory=False, prefetch_factor=4)
    test_data = th.utils.data.DataLoader(test_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=False, prefetch_factor=4)

    return train_data, test_data


class Dataset_task_2(Dataset):
    def __init__(self, file_path, itr=500, SEED=42, is_show=1, dataset_name="test", snr_db=None):
        super(Dataset_task_2, self).__init__()
        # Shuffle and Segmentation Dateset
        db = hdf5storage.loadmat(file_path + f'/X_{dataset_name}_prev.mat')[f'X_{dataset_name}_prev']
        db1 = hdf5storage.loadmat(file_path + f'/X_{dataset_name}.mat')[f'X_{dataset_name}']
        db2 = hdf5storage.loadmat(file_path + f'/imgs_{dataset_name}.mat')[f'imgs_{dataset_name}']
        db3 = hdf5storage.loadmat(file_path + f'/location_{dataset_name}.mat')[f'location_{dataset_name}']

        Batch_num = db.shape[0]
        np.random.seed(SEED)
        idx = np.arange(Batch_num)
        idx = idx[:itr]
        np.random.shuffle(idx)

        self.H_prev = torch.tensor(db, dtype=torch.complex128)
        self.H_target = torch.tensor(db1, dtype=torch.complex128)
        self.H_prev = LoadBatch(self.H_prev)
        self.H_target = LoadBatch(self.H_target)
        self.imgs = torch.tensor(db2, dtype=torch.float32).permute(0, 3, 1, 2)
        self.imgs = F.interpolate(self.imgs, size=(224, 224), mode='bilinear', align_corners=False) / 255.0
        self.location = torch.tensor(db3, dtype=torch.float32)

        # self.H_prev += torch.randn_like(self.H_prev) * torch.sqrt(torch.mean(self.H_prev**2) * 10**(-snr_db/10.0))
        
        self.H_prev = self.H_prev[idx, ...]
        self.H_target = self.H_target[idx, ...]
        self.imgs = self.imgs[idx, ...]
        self.location = self.location[idx, ...]

        if is_show:
            print('Dataset info: ')
            print(
                f'H_prev (input) shape: {self.H_prev.shape}\n'
                f'H_target (label) shape: {self.H_target.shape}\n'
                f'imgs (input) shape: {self.imgs.shape}\n'
                f'location (input) shape: {self.location.shape}\n'
            )

    def __getitem__(self, index):
        return {
            "H_prev": self.H_prev[index, ...].float(),
            "H_target": self.H_target[index, ...].float(),
            "rgb": self.imgs[index, ...].float(),
            "pos": self.location[index, ...].float(),
        }

    def __len__(self):
        return self.H_prev.shape[0]


def data_load_task_2(args):
    train_data = Dataset_task_2(args.data_path + '/Task2', itr=500, dataset_name="train", snr_db=args.snr_db)
    test_data = Dataset_task_2(args.data_path + '/Task2', itr=100, dataset_name="test", snr_db=args.snr_db)

    train_data = th.utils.data.DataLoader(train_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                          pin_memory=False, prefetch_factor=4)
    test_data = th.utils.data.DataLoader(test_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=False, prefetch_factor=4)

    return train_data, test_data


class Dataset_task_3(Dataset):
    def __init__(self, file_path, itr=1000, SEED=42, is_show=1, dataset_name="train"):
        super(Dataset_task_3, self).__init__()
        # Shuffle and Segmentation Dateset
        db = hdf5storage.loadmat(file_path + f'/H_{dataset_name}.mat')[f'H']
        db2 = hdf5storage.loadmat(file_path + f'/RGB_{dataset_name}.mat')[f'RGB']
        db3 = hdf5storage.loadmat(file_path + f'/depth_{dataset_name}.mat')[f'depth']
        Batch_num = db.shape[0]
        np.random.seed(SEED)
        idx = np.arange(Batch_num)
        np.random.shuffle(idx)
        idx = idx[:itr]

        self.H_full = torch.tensor(db, dtype=torch.complex128)
        self.H_full = LoadBatch(self.H_full)
        
        self.imgs = torch.tensor(db2, dtype=torch.float32).permute(0, 3, 1, 2)
        self.imgs = self.imgs[:, :, 428:652, 848:1072] / 255.0
        
        self.depth = torch.tensor(db3, dtype=torch.float32)
        self.depth = torch.flip(self.depth, dims=[1])
        self.depth = self.depth[:, 428:652, 848:1072].unsqueeze(1)
        
        self.H_full = self.H_full[idx, ...]
        self.imgs = self.imgs[idx, ...]
        self.depth = self.depth[idx, ...]

        if is_show:
            print('Dataset info: ')
            print(
                f'H_full (input) shape: {self.H_full.shape}\t'
                f'imgs (input) shape: {self.imgs.shape}\n'
                f'depth (output) shape: {self.depth.shape}\t'
            )
        
        if dataset_name == 'test':
            rgb = self.imgs[0].cpu()
            rgb = rgb.permute(1, 2, 0).numpy()
            plt.imsave('first_test_rgb.png', rgb)
            depth = self.depth[0].cpu().numpy().squeeze(0)
            depth = np.clip(depth, 0, 50)
            plt.figure()
            plt.imshow(depth, cmap='jet', vmin=0, vmax=50)
            plt.colorbar(label='Depth (m)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('first_test_depth.png', dpi=600, bbox_inches='tight')
            plt.close()
            np.savetxt('first_test_depth.txt', depth, fmt='%.6f')

    def __getitem__(self, index):
        return {
            "imgs": self.imgs[index, ...].float(),
            "H_full": self.H_full[index, ...].float(),
            "depth": self.depth[index, ...].float(),
        }

    def __len__(self):
        return self.H_full.shape[0]


def data_load_task_3(args):
    train_data = Dataset_task_3(args.data_path + '/Task3', itr=1000, dataset_name="train")
    test_data = Dataset_task_3(args.data_path + '/Task3', itr=200, dataset_name="test")

    train_data = th.utils.data.DataLoader(train_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                          pin_memory=False, prefetch_factor=4)
    test_data = th.utils.data.DataLoader(test_data, num_workers=8, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=False, prefetch_factor=4)

    return train_data, test_data


def data_load(args):
    if args.task_id == 1:
        train_data, test_data = data_load_task_1(args)
    if args.task_id == 2:
        train_data, test_data = data_load_task_2(args)
    if args.task_id == 3:
        train_data, test_data = data_load_task_3(args)
    return train_data, test_data
