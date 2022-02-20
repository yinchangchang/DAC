#!/usr/bin/env python
# coding=utf-8


import sys

import os
import sys
import time
import numpy as np
from sklearn import metrics
import random
import json
from collections import OrderedDict
from tqdm import tqdm


import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import data_loader
import model, loss

sys.path.append('../tools')
import parse, py_op

args = parse.args

model_dir = os.path.join(args.data_dir, 'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

def _cuda(tensor, is_tensor=True):
    if args.gpu:
        if is_tensor:
            return tensor.cuda()
        else:
            return tensor.cuda()
    else:
        return tensor


def train_eval_mortality_prediction(epoch, lr, mortatity_prediction, optimizer, enc_criterion, loader, best_metric, phase, table):
    if args.phase == 'train':
        mortatity_prediction.train()
    else:
        mortatity_prediction.eval()

    loss_list = []
    pred_list = []
    mort_list = []
    icustayid_risk_dict = dict()
    for data in tqdm(loader):
        data[:7] = [Variable(_cuda(x)) for x in data[:7]]
        collection_crt, collection_nxt, mask_crt, mask_nxt, action_crt, action_nxt, mortality = data[:7]
        # print('collection_crt', collection_crt.cpu().data.numpy().min(), collection_crt.cpu().data.numpy().max())
        risk_list = mortatity_prediction(collection_crt, action_crt) # [bs, 30, 25]
        loss = 0
        action_nxt_list = action_nxt
        for i in range(3):
            estimated_risk = risk_list[i]
            size = list(estimated_risk.size())
            risk = estimated_risk.view((-1, 7))
            action_nxt = action_nxt_list[:, :, i].view(-1)
            assert len(risk) == len(action_nxt)
            risk = risk[list(range(len(risk))), list(action_nxt.data.cpu().numpy())]
            risk = torch.nn.AdaptiveMaxPool1d(1)(risk.view(size[:2] + [1]).transpose(2, 1)).view(-1)
            # print(risk.size())
            assert len(risk) == len(mortality)
            mortality = mortality.view(-1)

            # positive
            ids = mortality > 0.5
            if len(risk[ids]):
                # print('--------------------------------------')
                # print(risk[ids].cpu().data.max(), risk[ids].cpu().data.min())
                # print(mortality[ids].cpu().data.max(), mortality[ids].cpu().data.min())
                loss += enc_criterion(risk[ids], mortality[ids])
            ids = mortality < 0.5
            if len(risk[ids]):
                # print('--------------------------------------')
                # print(risk[ids].cpu().data.max(), risk[ids].cpu().data.min())
                # print(mortality[ids].cpu().data.max(), mortality[ids].cpu().data.min())
                loss += enc_criterion(risk[ids], mortality[ids])

        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.data.cpu().numpy())
        mort_list += list(mortality.data.cpu().numpy().reshape(-1))
        pred_list += list(risk.data.cpu().numpy().reshape(-1))

        if phase == 'test':
            icustayid = data[-1].data.numpy().reshape(-1)
            risk_list = [r.data.cpu().numpy() for r in risk_list]
            for i in range(len(icustayid)):
                id = icustayid[i]
                # risk_list
                ri = [r[i] for r in risk_list]
                icustayid_risk_dict[str(int(id))] = ri
            # for id, ri in zip(icustayid, estimated_risk.data.cpu().numpy()):
            #     icustayid_risk_dict[str(int(id))] = ri
                # print(ri.shape)
    print('{:s} Epoch {:d} (lr {:0.5f})'.format(phase, epoch, lr))
    print('loss: {:3.4f} \t'.format(np.mean(loss_list))) 
    pred_list = np.array(pred_list).reshape(-1)
    mort_list = np.array(mort_list).reshape(-1)
    print(mort_list.shape)
    print(pred_list.shape)
    fpr, tpr, thresholds = metrics.roc_curve(mort_list, pred_list, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('auc', auc)
    if phase == 'valid':
        if auc > best_metric[1]:
            best_metric = [epoch, auc]
            torch.save(mortatity_prediction.state_dict(), os.path.join(model_dir, 'estimate_prediction.' + table))
        print('\t\t mortality prediction:   best epoch: {:d}     auc:{:0.4f}    '.format(best_metric[0], best_metric[1]))
        return best_metric
    if phase == 'test':
        return icustayid_risk_dict

def train_mortality_prediction(train_loader, valid_loader, table):

    mortatity_prediction= model.estimatePrediction_3(45, 128)
    enc_criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(mortatity_prediction.parameters(), lr=args.lr)
    # mortatity_prediction.load_state_dict(torch.load(os.path.join(model_dir, 'estimate_prediction.' + table)))
    mortatity_prediction.load_state_dict(torch.load(os.path.join(model_dir, 'estimate_prediction.' + table)))
    
    best_metric = [0, 0]
    for epoch in range(1, 1000):
        train_eval_mortality_prediction(epoch, args.lr, mortatity_prediction, optimizer, enc_criterion, train_loader, best_metric, 'train', table)
        best_metric = train_eval_mortality_prediction(epoch, args.lr, mortatity_prediction, optimizer, enc_criterion, valid_loader, best_metric, 'valid', table)
        if epoch > best_metric[0] + 15:
            break

    mortatity_prediction.load_state_dict(torch.load(os.path.join(model_dir, 'estimate_prediction.' + table)))
    icustayid_risk_dict = train_eval_mortality_prediction(epoch, args.lr, mortatity_prediction, optimizer, enc_criterion, train_loader, best_metric, 'test', table)
    _icustayid_risk_dict = train_eval_mortality_prediction(epoch, args.lr, mortatity_prediction, optimizer, enc_criterion, valid_loader, best_metric, 'test', table)
    icustayid_risk_dict.update(_icustayid_risk_dict)
    # py_op.mywritejson(os.path.join(args.file_dir, 'icustayid_risk_dict.json'), icustayid_risk_dict)
    torch.save(icustayid_risk_dict, os.path.join(args.file_dir, table + '.icustayid_risk_dict.ckpt'))

def estimated_mortality_for_dataset(table):
    icustayid_split_dict = py_op.myreadjson(os.path.join(args.file_dir, table + '.icustayid_split_dict.json'))
    icustayid_train = icustayid_split_dict['icustayid_train']
    icustayid_valid = icustayid_split_dict['icustayid_valid']
    # id risk
    dataset = data_loader.DataBowl(args, icustayid_train, phase='train', dataset=table)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dataset = data_loader.DataBowl(args, icustayid_valid, phase='valid', dataset=table)
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_mortality_prediction(train_loader, valid_loader, table)

def main():
    estimated_mortality_for_dataset('mechvent')



if __name__ == '__main__':
    main()
