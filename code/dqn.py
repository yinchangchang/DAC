#!/usr/bin/env python
# coding=utf-8


import sys

import os
import sys
import time
import random
import json
from collections import OrderedDict
from tqdm import tqdm


import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader

import data_loader
import model, loss

sys.path.append('../tools')
import parse, py_op
import numpy as np
from sklearn import metrics

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

def collect_gt_prob(decoded_nxt, action_nxt, mask_nxt, mortality, estimated_mortality, gamma=0.99):
    probs_list = []
    rewas_list = []
    preds_list = []
    trues_list = []
    morts_list = []
    for dn, an, mn, mo, em in zip(decoded_nxt.data.cpu().numpy(), action_nxt.data.cpu().numpy(), \
            mask_nxt.data.cpu().numpy(), mortality.data.cpu().numpy(), estimated_mortality.data.cpu().numpy()):
        prob_list = []
        rewa_list = []
        pred_list = []
        true_list = []
        mort_list = []
        mo = 2 * (0.5 - mo[0])
        for d, a, m, e in zip(dn, an, mn, em):
            if m < 0.5:
                rewa_list.append(0)
                # mo = mo * gamma
                prob_list.append(d[int(a)])
                true_list.append(int(a))
                pred_list.append(np.argmax(d))
                # mort_list.append(e[int(np.argmax(d))])
                mort_list.append(0.2)
        if len(rewa_list) > 0:
            rewa_list[-1] = mo * 15
        probs_list.append(prob_list)
        rewas_list.append(rewa_list)
        preds_list.append(pred_list)
        trues_list.append(true_list)
        morts_list.append(mort_list)
    return probs_list, rewas_list, preds_list, trues_list, morts_list

def train_eval_action_imitation(epoch, lr, action_imitation, optimizer, enc_criterion, loader, best_metric, phase, gamma=0.99):
    if args.phase == 'train':
        action_imitation.train()
    else:
        action_imitation.eval()

    loss_list = []
    sum_correct, sum_valid = 0, 0
    probs_list, rewas_list, preds_list, trues_list, morts_list = [], [], [], [], []
    for data in tqdm(loader):
        data[:-1] = [Variable(_cuda(x)) for x in data[:-1]]
        if phase == 'train' and args.use_ci:
            for i in range(len(data) - 1):
                if args.use_ci == 3:
                    assert data[i].size(1) == 4
                else:
                    assert data[i].size(1) == 2
                size = list(data[i].size())
                new_size = [size[0] * size[1]] + size[2:]
                data[i] = data[i].view(new_size)
        collection_crt, collection_nxt, mask_crt, mask_nxt, action_crt, action_nxt, mortality, reward, estimated_mortality = data[:9]
        q_value, decoded_nxt = action_imitation(collection_crt, action_crt)
        # decoded_nxt = F.softmax(q_value, 2)
        # print(q_value.size())
        # print(decoded_nxt.data.cpu().numpy().sum(2))

        # print(reward.size(), mask_crt.size())
        # print('reward crt', (reward * mask_crt[:,:,0]).max(), (reward * mask_crt[:,:,0]).min())
        # print('reward nxt', (reward * mask_nxt[:,:,0]).max(), (reward * mask_nxt[:,:,0]).min())
        # print(err)
        # print(q_value.size(), mask_nxt.size())
        q_value = q_value * (1 - mask_nxt[:,:,:25])
        # print(q_value.size(), action_nxt.size())

        q_size = list(q_value.size())
        q_value = q_value.view(-1, q_size[2])
        action_nxt = action_nxt.view(-1)
        current_q_value = q_value[list(range(len(q_value))), list(action_nxt.data.cpu().numpy().astype(np.int32))].view(q_size[:2])
        q_value = q_value.view(q_size)
        action_nxt = action_nxt.view([q_size[0], -1])

        # print(q_value[0, :3], action_nxt[0, :3])
        # print(current_q_value[0, :3])
        # print('current_q_value', current_q_value.size(), '[-, 30]')
        next_q_value = q_value.detach().max(2)[0]
        # print(q_value.detach().max(2)[0])
        # print(q_value.detach().max(2)[1])
        # print('next_q_value', next_q_value.size(), '[-, 30]')
        update_q_value = current_q_value[:, :-1]
        # current_q_value = current_q_value[:, 1:]
        # current_reward = reward[:, :-1]
        # next_q_value = next_q_value[:, :-1]

        current_q_value = current_q_value[:, :-1]
        current_reward = reward[:, 1:]
        next_q_value = next_q_value[:, 1:]

        target_q_value = current_reward + (gamma * next_q_value)
        bellman_error = target_q_value - current_q_value
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        d_error = clipped_bellman_error * -1.0


        ps_list, rs_list, pr_list, ts_list, ms_list = collect_gt_prob(decoded_nxt, action_nxt, mask_nxt[:, :, 0], mortality, estimated_mortality)
        probs_list += ps_list
        rewas_list += rs_list
        preds_list += pr_list
        trues_list += ts_list
        morts_list += ms_list
        # print(len(probs_list), len(rewas_list))

        loss, n_correct, n_valid = enc_criterion(decoded_nxt, action_nxt, mask_nxt[:, :, 0], reward)
        # print(n_correct, n_valid)
        sum_correct += n_correct
        sum_valid += n_valid
        if phase == 'train':
            optimizer.zero_grad()
            if args.loss == 'dqn':
                # (loss*10).backward(retain_graph=True)
                current_q_value.backward(d_error.data)
                # loss.backward(retain_graph=True)
                # update_q_value.backward(d_error.data * 0.1)
            else:
                loss.backward()
            optimizer.step()

        loss_list.append(loss.data.cpu().numpy())
    print('{:s} Epoch {:d} (lr {:0.5f})'.format(phase, epoch, lr))
    print('loss: {:3.4f} \t'.format(np.mean(loss_list))) 
    acc = float(sum_correct) / max(sum_valid, 1)
    print('accuracy: {:3.4f} \t'.format(acc))
    wis = py_op.compute_wis(probs_list, rewas_list)
    jaccard = py_op.compute_jaccard(preds_list, trues_list)
    em = py_op.compute_estimated_mortality(morts_list)
    print('wis: {:3.4f} \t'.format(wis))
    print('jaccard: {:3.4f} \t'.format(jaccard))
    print('estimated mortality: {:3.4f} \t'.format(em))
    if phase == 'valid':
        # if acc > best_metric[2] and wis > 7:
        if wis > best_metric[1] and acc > 0.15 and wis > 8:
            best_metric = [epoch, wis, acc, jaccard, em]
            # torch.save(action_imitation.state_dict(), os.path.join(model_dir, 'action_imitation_{:d}_{:1.2f}_{:1.3f}'.format(args.use_ci, acc, em)))
            torch.save(action_imitation.state_dict(), os.path.join(model_dir, 'action_imitation_{:d}_{:1.1f}_{:1.3f}_{:1.3f}'.format(args.use_ci, wis, acc, em)))
        '''
        if args.loss == 'dqn' and em < best_metric[4] and wis > best_metric[1]:
            best_metric = [epoch, wis, acc, jaccard, em]
            torch.save(action_imitation.state_dict(), os.path.join(model_dir, 'action_imitation_{:d}_{:d}_{:1.3f}'.format(args.use_ci, int(wis), em)))
        elif args.loss == 'dqn' and em < best_metric[4] and wis > 12:
            best_metric = [epoch, wis, acc, jaccard, em]
            torch.save(action_imitation.state_dict(), os.path.join(model_dir, 'action_imitation_{:d}_{:d}_{:1.3f}'.format(args.use_ci, int(wis), em)))
        elif args.loss != 'dqn' and wis > best_metric[1]:
            best_metric = [epoch, wis, acc, jaccard, em]
            torch.save(action_imitation.state_dict(), os.path.join(model_dir, 'action_imitation_{:d}_{:d}_{:1.3f}'.format(args.use_ci, int(wis), em)))
        '''
        print('\t\t action imitation:   best epoch: {:d}     wis:{:0.4f}    acc:{:0.4f}    jaccard:{:1.4f}    em:{:1.4f}'.format(best_metric[0], best_metric[1], best_metric[2], best_metric[3], best_metric[4]))
        # print(type(probs_list), type(rewas_list))
        return best_metric

def train_action_imitation(train_loader, valid_loader):

    # autoencoder = model.autoEncoder(30, 128)
    # autoencoder.load_state_dict(torch.load(os.path.join(model_dir, 'autoencoder')))
    action_imitation = model.actionImitation(30, 128)
    qloss = loss.QLoss()
    closs  = loss.ClassifyLoss()
    optimizer = torch.optim.RMSprop(action_imitation.parameters(), lr=args.lr)
    
    best_metric = [0, 0, 0, 0, 1]
    for epoch in range(1, 10):
        enc_criterion = closs
        train_eval_action_imitation(epoch, args.lr, action_imitation, optimizer, enc_criterion, train_loader, best_metric, 'train')
        best_metric = train_eval_action_imitation(epoch, args.lr, action_imitation, optimizer, enc_criterion, valid_loader, best_metric, 'valid')


def main():
    if args.phase == 'train':
        icustayid_split_dict = py_op.myreadjson(os.path.join(args.file_dir, 'mimic.icustayid_split_dict.json'))
        icustayid_train = icustayid_split_dict['icustayid_train']
        icustayid_valid = icustayid_split_dict['icustayid_valid']
        if args.use_ci:
            dataset = data_loader.CIDataBowl(args, icustayid_train, phase='train', dataset='mimic')
        else:
            dataset = data_loader.DataBowl(args, icustayid_train, phase='train')
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)


        # icustayid_split_dict = py_op.myreadjson(os.path.join(args.file_dir, 'ast.icustayid_split_dict.json'))
        # icustayid_train = icustayid_split_dict['icustayid_train']
        # icustayid_valid = icustayid_split_dict['icustayid_valid']
        # dataset = data_loader.DataBowl(args, icustayid_train + icustayid_valid, phase='valid', dataset='ast')
        dataset = data_loader.DataBowl(args, icustayid_valid, phase='valid', dataset='mimic')
        valid_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        train_action_imitation(train_loader, valid_loader)
    else:
        if 0:
            icustayid_split_dict = py_op.myreadjson(os.path.join(args.file_dir, 'ast.icustayid_split_dict.json'))
            icustayid_train = icustayid_split_dict['icustayid_train']
            icustayid_valid = icustayid_split_dict['icustayid_valid']
            dataset = data_loader.DataBowl(args, icustayid_train + icustayid_valid, phase='valid', dataset='ast')
        else:
            icustayid_split_dict = py_op.myreadjson(os.path.join(args.file_dir, 'mimic.icustayid_split_dict.json'))
            icustayid_test = icustayid_split_dict['icustayid_test']
            dataset = data_loader.DataBowl(args, icustayid_test, phase='valid', dataset='mimic')
        valid_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        action_imitation = model.actionImitation(30, 128)
        action_imitation.load_state_dict(torch.load(args.resume))
        enc_criterion = loss.ClassifyLoss()
        optimizer = torch.optim.RMSprop(action_imitation.parameters(), lr=args.lr)
        best_metric = [0, 0, 0, 0, 1]
        train_eval_action_imitation(0, args.lr, action_imitation, optimizer, enc_criterion, valid_loader, best_metric, 'valid')



if __name__ == '__main__':
    main()
