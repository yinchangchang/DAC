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
from torch.nn import functional as F
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

def preprocess_data(data, phase):
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
    return data

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
                mort_list.append(e[int(np.argmax(d))])
        if len(rewa_list) > 0:
            rewa_list[-1] = mo * 15
        probs_list.append(prob_list)
        rewas_list.append(rewa_list)
        preds_list.append(pred_list)
        trues_list.append(true_list)
        morts_list.append(mort_list)
    return probs_list, rewas_list, preds_list, trues_list, morts_list

def train_eval_actor(epoch, lr, autoencoder, mortatity_prediction, action_imitation, actor, optimizer, enc_criterion, loader, best_metric, phase='train', gamma=0.99):
    if args.phase == 'train':
        actor.train()
    else:
        actor.eval()
    action_imitation.eval()
    autoencoder.eval()
    mortatity_prediction.eval()

    loss_list = []
    sum_correct, sum_valid = 0, 0
    probs_list, rewas_list, preds_list, trues_list, morts_list = [], [], [], [], []
    for data in tqdm(loader):
        data = preprocess_data(data, phase)
        collection_crt, collection_nxt, mask_crt, mask_nxt, action_crt, action_nxt, mortality_90d, reward, estimated_mortality, mortality_48h, icustayid = data 

        hidden_state, _, _ = autoencoder(collection_crt)
        hidden_state = hidden_state.detach()
        reward = mortatity_prediction(hidden_state).detach()
        _, prob_imitation = action_imitation(hidden_state, action_crt)
        prob_imitation = prob_imitation.detach()
        _, prob = actor(hidden_state)
        # _, prob = actor(collection_crt)
        decoded_nxt = prob

        reward_delta = reward.detach()
        reward_delta[:, :-1] = reward[:, 1:] - reward[:, :-1]
        reward_delta[:, -1] = 0
        reward_delta = reward_delta.view(-1)
        reward_new = reward.detach()
        reward_new[:, :-1] = reward[:, 1:] 
        reward_new[:, -1] = 0
        reward_new = reward_new.view(-1)
        reward = reward_new



        reward = reward.view(-1)
        prob = prob.view(-1, 25)
        prob_imitation = prob_imitation.view(-1, 25)
        mask = mask_nxt[:, :, 0].view(-1)

        # reward[prob_imitation < 1.0/100] = 1

        # print('reward', reward.size())
        # print('prob', prob.size())
        # print('prob_imitation', prob_imitation.size())
        # print(err)

        idx = mask < 0.5
        reward_delta = reward_delta[idx]
        reward = reward[idx]
        prob = prob[idx]
        prob_init = prob
        an = action_nxt.view((-1))[idx]
        prob = prob[list(range(len(prob))), list(an.data.cpu().numpy())]
        # print('reward', reward.size())
        # print('prob', prob.size())

        th= 2
        # idx = reward < th
        idx = reward < th
        _reward = reward[idx] - th
        _prob = prob[idx]
        _reward = _reward.view((-1, 1, 1)) 
        _prob = _prob.view((-1, 1, 1))
        r1 = torch.bmm(_reward, _prob)
        loss = r1.sum() / len(r1) 

        '''

        idx = reward > th
        _reward = reward_delta[idx]
        _prob = prob[idx]
        _reward = _reward.view((-1, 1, 1)) 
        _prob = _prob.view((-1, 1, 1))
        r2 = torch.bmm(_reward, _prob)
        loss += r2.sum() / len(r2)
        # '''


        ps_list, rs_list, pr_list, ts_list, ms_list = collect_gt_prob(decoded_nxt, action_nxt, mask_nxt[:, :, 0], mortality_90d, estimated_mortality)
        probs_list += ps_list
        rewas_list += rs_list
        preds_list += pr_list
        trues_list += ts_list
        morts_list += ms_list
        # print(len(probs_list), len(rewas_list))

        # print(n_correct, n_valid)
        # print(an.data.cpu().numpy().shape, prob_init.data.cpu().numpy().shape)
        gt = an.data.cpu().numpy()
        pd = prob_init.data.cpu().numpy().argmax(1).reshape(-1)

        idx = gt > 0
        gt = gt[idx]
        pd = pd[idx]

        n_correct = ( gt==pd ).sum()
        n_valid = len(gt)
        # print(n_valid, n_correct)
        sum_correct += n_correct
        sum_valid += n_valid
        # loss = r1.sum() / len(r1) 
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss.data.cpu().numpy().mean())
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
        # if acc > best_metric[2]:
        if wis > best_metric[1]:
            best_metric = [epoch, wis, acc, jaccard, em]
            # torch.save(actor.state_dict(), os.path.join(model_dir, 'actor_{:d}_{:1.2f}_{:1.3f}'.format(args.use_ci, acc, em)))
            torch.save(actor.state_dict(), os.path.join(model_dir, 'actor_{:d}_{:1.1f}_{:1.3f}_{:1.3f}'.format(args.use_ci, wis, acc, em)))
        '''
        if args.loss == 'dqn' and em < best_metric[4] and wis > best_metric[1]:
            best_metric = [epoch, wis, acc, jaccard, em]
            torch.save(actor.state_dict(), os.path.join(model_dir, 'actor_{:d}_{:d}_{:1.3f}'.format(args.use_ci, int(wis), em)))
        elif args.loss == 'dqn' and em < best_metric[4] and wis > 12:
            best_metric = [epoch, wis, acc, jaccard, em]
            torch.save(actor.state_dict(), os.path.join(model_dir, 'actor_{:d}_{:d}_{:1.3f}'.format(args.use_ci, int(wis), em)))
        elif args.loss != 'dqn' and wis > best_metric[1]:
            best_metric = [epoch, wis, acc, jaccard, em]
            torch.save(actor.state_dict(), os.path.join(model_dir, 'actor_{:d}_{:d}_{:1.3f}'.format(args.use_ci, int(wis), em)))
        '''
        print('\t\t action imitation:   best epoch: {:d}     wis:{:0.4f}    acc:{:0.4f}    jaccard:{:1.4f}    em:{:1.4f}'.format(best_metric[0], best_metric[1], best_metric[2], best_metric[3], best_metric[4]))
        # print(type(probs_list), type(rewas_list))
        return best_metric

def train_eval_action_imitation(epoch, lr, autoencoder, action_imitation, optimizer, enc_criterion, loader, best_metric, phase, gamma=0.99):
    if args.phase == 'train':
        action_imitation.train()
    else:
        action_imitation.eval()
    autoencoder.eval()

    loss_list = []
    sum_correct, sum_valid = 0, 0
    probs_list, rewas_list, preds_list, trues_list, morts_list = [], [], [], [], []
    for data in tqdm(loader):
        data = preprocess_data(data, phase)
        collection_crt, collection_nxt, mask_crt, mask_nxt, action_crt, action_nxt, mortality_90d, reward, estimated_mortality, mortality_48h, icustayid = data 
        hidden_state, _, _ = autoencoder(collection_crt)
        hidden_state = hidden_state.detach()
        _, decoded_nxt = action_imitation(hidden_state, action_crt)
        ps_list, rs_list, pr_list, ts_list, ms_list = collect_gt_prob(decoded_nxt, action_nxt, mask_nxt[:, :, 0], mortality_90d, estimated_mortality)
        probs_list += ps_list
        rewas_list += rs_list
        preds_list += pr_list
        trues_list += ts_list
        morts_list += ms_list

        loss, n_correct, n_valid = enc_criterion(decoded_nxt, action_nxt, mask_nxt[:, :, 0], reward)
        sum_correct += n_correct
        sum_valid += n_valid
        if phase == 'train':
            optimizer.zero_grad()
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
        # if acc > best_metric[2]:
        if wis > best_metric[1]:
            best_metric = [epoch, wis, acc, jaccard, em]
            # torch.save(action_imitation.state_dict(), os.path.join(model_dir, 'action_imitation_{:d}_{:1.2f}_{:1.3f}'.format(args.use_ci, acc, em)))
            torch.save(action_imitation.state_dict(), os.path.join(model_dir, 'action_imitation'))
        print('\t\t action imitation:   best epoch: {:d}     wis:{:0.4f}    acc:{:0.4f}    jaccard:{:1.4f}    em:{:1.4f}'.format(best_metric[0], best_metric[1], best_metric[2], best_metric[3], best_metric[4]))
        # print(type(probs_list), type(rewas_list))
        return best_metric

def train_action_imitation(train_loader, valid_loader):

    autoencoder = model.autoEncoder(45, 128)
    autoencoder.load_state_dict(torch.load(os.path.join(model_dir, 'autoencoder')))
    action_imitation = model.actionImitation(45, 128)
    enc_criterion = loss.ClassifyLoss()
    optimizer = torch.optim.RMSprop(action_imitation.parameters(), lr=args.lr)
    
    best_metric = [0, 0, 0, 0, 1]
    for epoch in range(1, 40):
        train_eval_action_imitation(epoch, args.lr, autoencoder, action_imitation, optimizer, enc_criterion, train_loader, best_metric, 'train')
        best_metric = train_eval_action_imitation(epoch, args.lr, autoencoder, action_imitation, optimizer, enc_criterion, valid_loader, best_metric, 'valid')

def train_eval_ae_epoch(epoch, lr, autoencoder, optimizer, enc_criterion, loader, best_metric, phase):
    if args.phase == 'train':
        autoencoder.train()
    else:
        autoencoder.eval()

    loss_list = []
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
        collection_crt, collection_nxt, mask_crt, mask_nxt, action_crt, action_nxt = data[:6]
        encoded, decoded_crt, decoded_nxt = autoencoder(collection_crt)
        loss = enc_criterion(decoded_crt, collection_crt, mask_crt) + enc_criterion(decoded_nxt, collection_nxt, mask_nxt)
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.data.cpu().numpy())
    print('{:s} Epoch {:d} (lr {:0.5f})'.format(phase, epoch, lr))
    print('loss: {:3.4f} \t'.format(np.mean(loss_list))) 
    if phase == 'valid':
        if np.mean(loss_list) < best_metric[1]:
            best_metric = [epoch, np.mean(loss_list)]
            torch.save(autoencoder.state_dict(), os.path.join(model_dir, 'autoencoder'))
        print('\t\t autoencoder:   best epoch: {:d}     \t best metric:{:0.8f}'.format(best_metric[0], best_metric[1]))
        return best_metric


def train_autoencoder(train_loader, valid_loader):

    autoencoder = model.autoEncoder(45, 128)
    # autoencoder.load_state_dict(torch.load(os.path.join(model_dir, 'autoencoder')))
    enc_criterion = loss.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    
    best_metric = [0, 10e10]
    for epoch in range(1, 50):
        train_eval_ae_epoch(epoch, args.lr, autoencoder, optimizer, enc_criterion, train_loader, best_metric, 'train')
        best_metric = train_eval_ae_epoch(epoch, args.lr, autoencoder, optimizer, enc_criterion, valid_loader, best_metric, 'valid')

def train_eval_critic(epoch, lr, autoencoder, mortatity_prediction, optimizer, enc_criterion, loader, best_metric, phase):
    if args.phase == 'train':
        mortatity_prediction.train()
    else:
        mortatity_prediction.eval()
    autoencoder.eval()

    loss_list = []
    pred_list = []
    mort_list = []
    icustayid_risk_dict = dict()
    for data in tqdm(loader):
        data = preprocess_data(data, phase)
        collection_crt, collection_nxt, mask_crt, mask_nxt, action_crt, action_nxt, mortality_90d, reward, estimated_mortality, mortality_48h, icustayid = data 

        hidden_state, _, _ = autoencoder(collection_crt)
        hidden_state = hidden_state.detach()
        risk = mortatity_prediction(hidden_state)
        size = list(mortality_48h.size())

        # the mortality rate with corresponding action
        # risk = risk.view((-1, 25))
        # idx = action_nxt.view(-1).data.cpu().numpy().astype(np.int32)
        # risk = risk[list(range(len(risk))), list(idx)].view(-1)
        risk = risk.view(-1)
        mask = mask_nxt[:,:,0].view(-1)
        mortality_48h = mortality_48h.view(-1)
        # print('mask', mask.size())
        # print('risk', risk.size())
        # print('mortality_48h', mortality_48h.size())
        # print(err)

        risk = risk[mask<0.5]
        mortality_48h = mortality_48h[mask<0.5]

        # positive 
        loss = 0
        idx = mortality_48h>0.5
        _risk = risk[idx]
        _label = mortality_48h[idx]
        if len(_risk):
            loss += enc_criterion(_risk, _label)
        # negative
        idx = mortality_48h<0.5
        _risk = risk[idx]
        _label = mortality_48h[idx]
        if len(_risk):
            loss += enc_criterion(_risk, _label)

        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.data.cpu().numpy())

        mort_list += list(mortality_48h.data.cpu().numpy().reshape(-1))
        pred_list += list(risk.data.cpu().numpy().reshape(-1))

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
            torch.save(mortatity_prediction.state_dict(), os.path.join(model_dir, 'mortatity_prediction'))
        print('\t\t mortality prediction:   best epoch: {:d}     auc:{:0.4f}    '.format(best_metric[0], best_metric[1]))
        return best_metric

def train_critic(train_loader, valid_loader):

    autoencoder = model.autoEncoder(45, 128)
    autoencoder.load_state_dict(torch.load(os.path.join(model_dir, 'autoencoder')))
    mortatity_prediction= model.Critic_s(45, 128)
    enc_criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(mortatity_prediction.parameters(), lr=args.lr)
    
    best_metric = [0, 0]
    for epoch in range(1, 45):
        train_eval_critic(epoch, args.lr, autoencoder, mortatity_prediction, optimizer, enc_criterion, train_loader, best_metric, 'train')
        best_metric = train_eval_critic(epoch, args.lr, autoencoder, mortatity_prediction, optimizer, enc_criterion, valid_loader, best_metric, 'valid')

def train_actor(train_loader, valid_loader):

    autoencoder = model.autoEncoder(45, 128)
    autoencoder.load_state_dict(torch.load(os.path.join(model_dir, 'autoencoder')))
    mortatity_prediction= model.Critic_s(45, 128)
    mortatity_prediction.load_state_dict(torch.load(os.path.join(model_dir, 'mortatity_prediction')))
    action_imitation = model.actionImitation(45, 128)
    action_imitation.load_state_dict(torch.load(os.path.join(model_dir, 'action_imitation')))
    enc_criterion = torch.nn.BCELoss()
    actor = model.Actor(45, 128)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    
    best_metric = [0, 0]
    for epoch in range(1, 300):
        train_eval_actor(epoch, args.lr, autoencoder, mortatity_prediction, action_imitation, actor, optimizer, enc_criterion, train_loader, best_metric, 'train')
        best_metric = train_eval_actor(epoch, args.lr, autoencoder, mortatity_prediction, action_imitation, actor, optimizer, enc_criterion, valid_loader, best_metric, 'valid')

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

        # train_autoencoder(train_loader, valid_loader)
        # train_critic(train_loader, valid_loader)
        # train_action_imitation(train_loader, valid_loader)
        train_actor(train_loader, valid_loader)
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
        action_imitation = model.actionImitation(45, 128)
        action_imitation.load_state_dict(torch.load(args.resume))
        enc_criterion = loss.ClassifyLoss()
        optimizer = torch.optim.RMSprop(action_imitation.parameters(), lr=args.lr)
        best_metric = [0, 0, 0, 0, 1]
        train_eval_action_imitation(0, args.lr, action_imitation, optimizer, enc_criterion, valid_loader, best_metric, 'valid')



if __name__ == '__main__':
    main()
