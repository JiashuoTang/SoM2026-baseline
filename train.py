# coding=utf-8
import os
import torch
from torch.optim import AdamW, SGD, Adam
import random
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import time
import collections
import torch.nn as nn
from sklearn.metrics import f1_score


class TrainLoop:
    def __init__(self, args, writer, model, optimizer, scheduler, train_data, test_data, device, early_stop=5, test_fre=1):
        self.args = args
        self.writer = writer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.log_interval = args.log_interval
        self.best_nmse_random = 1e9
        self.warmup_steps = 5
        self.min_lr = args.min_lr
        self.best = 1e9
        self.early_stop = early_stop
        self.test_fre = test_fre
        self.criterion_t1 = nn.CrossEntropyLoss().to(device)
        self.criterion_t2 = nn.MSELoss().to(device)
        self.criterion_t3 = nn.L1Loss().to(device)


    def Train_iter(self, train_data, mask_ratio, mask_strategy, seed=None, dataset='', mode='backward', is_output_mat=False, task_id=1):
        loss_list = []
        abs_error_sum = 0.0
        squared_error_sum = 0.0
        abs_rel_sum = 0.0
        delta1_count = 0
        valid_count = 0
        sample_count = 0
        # start time
        for _, batch in enumerate(train_data):
            # print(batch.shape)
            if mode == 'backward':
                self.optimizer.zero_grad()
            pred = self.model_forward(batch, mask_ratio, mask_strategy, seed=seed, data=dataset, mode='forward', task_id=task_id)
            sample_count += pred.shape[0]
            # print(pred.shape, batch['label'].shape)
            if mode == 'backward':
                if task_id == 1:
                    loss = self.criterion_t1(pred, batch['label'].squeeze())
                elif task_id == 2:
                    loss = self.criterion_t2(pred[..., 32:], batch['H_target'][..., 32:])
                elif task_id == 3:
                    valid = (batch['depth'] > 0) & (batch['depth'] < 50)
                    loss = self.criterion_t3(pred[valid], batch['depth'][valid])

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                loss_list.append(loss.item())
            else:
                if task_id == 1:
                    pred_classes = pred.argmax(dim=1)  # [B]
                    # cal acc
                    # correct = (pred_classes == batch['label'].squeeze()).float()  # [B], 0/1
                    # acc = correct.mean().item()
                    # result = 1 - acc
                    # cal F1
                    f1_macro = f1_score(batch['label'].squeeze().detach().cpu(), pred_classes.detach().cpu(), average='macro')
                    result = 1 - f1_macro
                    loss_list.append(result)
                elif task_id == 2:
                    # cal NMSE
                    pred = pred[..., 32:]
                    target = batch['H_target'][..., 32:]
                    diff = torch.abs(pred - target) ** 2
                    power = torch.abs(target) ** 2
                    nmse = torch.sum(diff, dim=tuple(range(1, diff.ndim))) / torch.sum(power, dim=tuple(range(1, power.ndim)))
                    loss_list.extend(nmse.detach().cpu().tolist())
                elif task_id == 3:
                    depth = batch['depth']
                    valid = (depth > 0) & (depth < 50)
                    pred_valid = pred[valid]
                    depth_valid = depth[valid]
                    abs_error = torch.abs(pred_valid - depth_valid)
                    abs_error_sum += abs_error.sum().item()
                    squared_error_sum += torch.square(pred_valid - depth_valid).sum().item()
                    abs_rel_sum += (abs_error / depth_valid).sum().item()
                    ratio = torch.maximum(pred_valid / depth_valid, depth_valid / pred_valid,)
                    delta1_count += (ratio < 1.25).sum().item()
                    valid_count += depth_valid.numel()
        if mode != 'backward' and task_id == 3:
            return {
                'mae': abs_error_sum / valid_count,
                'rmse': math.sqrt(squared_error_sum / valid_count),
                'abs_rel': abs_rel_sum / valid_count,
                'delta1': delta1_count / valid_count,
            }
        return np.mean(np.array(loss_list))


    def Train(self, seed=None, mask_ratio=None, mask_strategy=None, task_id=1):
        nmse_list = []
        for epoch in range(self.args.epochs):
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    own_params = list(module.parameters(recurse=False))
                    if own_params and not any(p.requires_grad for p in own_params):
                        module.eval()
            loss_iter = self.Train_iter(
                self.train_data, mask_ratio=mask_ratio, mask_strategy=mask_strategy, 
                seed=seed, dataset='None', mode='backward', task_id=task_id
            )
            print(f'Epoch [{epoch + 1}/{self.args.epochs}] Training Loss: {loss_iter}')
            nmse_list.append(loss_iter)
            if epoch % self.test_fre == 0:
                with torch.no_grad():
                    self.model.eval()
                    result = self.Train_iter(
                        self.test_data, mask_ratio=mask_ratio, mask_strategy=mask_strategy,
                        seed=seed, dataset='None', mode='test', task_id=task_id
                    )
                    metric = result['mae'] if task_id == 3 else result
                    if metric < self.best:
                        is_break = self.best_model_save(epoch, metric, result)
                    print(f'Epoch [{epoch + 1}/{self.args.epochs}] Testing Loss: {result}', end=' || ')
                    print(' Test_Result_best:{}\n'.format(self.best))


    def Test(self, seed=None, mask_ratio=None, mask_strategy=None, task_id=1):
        with torch.no_grad():
            self.model.eval()
            result = self.Train_iter(
                self.test_data, mask_ratio=mask_ratio, mask_strategy=mask_strategy,
                seed=seed, dataset='None', mode='test', task_id=task_id
            )
        print(result)
        with open(self.args.model_path+'result.txt', 'w') as f:
            f.write('Test result:{}\n'.format(result))


    def best_model_save(self, step, metric, key_result):
        self.early_stop = 0
        torch.save(self.model.state_dict(), self.args.model_path + 'model_save/model_best.pkl')
        self.best = metric
        print('\nTest best:{}\n'.format(self.best))
        print(str(key_result) + '\n')
        with open(self.args.model_path+'result.txt', 'w') as f:
            f.write('epoch:{}, best:{}\n'.format(step, self.best))
            f.write(str(key_result) + '\n')
        with open(self.args.model_path+'result_all.txt', 'a') as f:
            f.write('epoch:{}, best:{}\n'.format(step, self.best))
            f.write(str(key_result) + '\n')
        return 'save'


    def run(self, seed=None, mask_ratio=None, mask_strategy=None):
        if self.args.task == 'Test':
            self.Test(task_id=self.args.task_id)
        else:
            self.Train(task_id=self.args.task_id)
            
    
    def model_forward(self, batch, mask_ratio, mask_strategy, seed=None, data=None, mode='backward', task_id=1):
        for value, key in batch.items():
            batch[value] = key.to(self.device)
        if task_id == 1:
            pred = self.model(
                batch['H_full'],
                mask_ratio=0,
                mask_strategy='fre',
                seed=seed,
                data=data, 
                task_id=task_id,
            )
        elif task_id == 2:
            pred = self.model(
                batch['H_prev'],
                rgb=batch['rgb'],
                pos=batch['pos'],
                mask_ratio=0.5,
                mask_strategy='fre',
                seed=seed,
                data=data, 
                task_id=task_id,
            )
        elif task_id == 3:
            pred = self.model(
                batch['H_full'],
                rgb=batch['imgs'],
                mask_ratio=0,
                mask_strategy='fre',
                seed=seed,
                data=data, 
                task_id=task_id,
            )
        return pred
