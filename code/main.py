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
short_loss = loss.ShortLoss_3()

model_dir = os.path.join(args.data_dir, 'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

try:
    icustayid_risk_dict = torch.load(os.path.join(args.file_dir, 'mechvent.icustayid_risk_dict.ckpt'))
except:
    icustayid_risk_dict = dict()
icustayid_risk_dict = torch.load(os.path.join(args.file_dir, 'mechvent.icustayid_risk_dict.ckpt'))

def _cuda(tensor, is_tensor=True):
    if args.gpu:
        if is_tensor:
            return tensor.cuda()
        else:
            return tensor.cuda()
    else:
        return tensor

def train_eval_ae_epoch(epoch, lr, autoencoder, optimizer, enc_criterion, loader, best_metric, phase):
    if args.phase == 'train':
        autoencoder.train()
    else:
        autoencoder.eval()

    loss_list = []
    for data in tqdm(loader):
        data = [Variable(_cuda(x)) for x in data]
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

def train_eval_state_prediction(epoch, lr, autoencoder, state_prediction, optimizer, enc_criterion, loader, best_metric, phase):
    if args.phase == 'train':
        state_prediction.train()
    else:
        state_prediction.eval()
    autoencoder.eval()

    loss_list = []
    for data in tqdm(loader):
        data = [Variable(_cuda(x)) for x in data]
        collection_crt, collection_nxt, mask_crt, mask_nxt, action_crt, action_nxt = data[:6]
        hidden_state, _, _ = autoencoder(collection_crt)
        hidden_state_nxt, decoded_nxt = state_prediction(hidden_state, action_nxt)

        decoded_nxt = decoded_nxt.view(-1)
        collection_nxt = collection_nxt.view(-1)
        mask_nxt = mask_nxt.view(-1)
        assert len(decoded_nxt) == len(collection_nxt) == len(mask_nxt)

        ids = mask_nxt < 0
        loss = enc_criterion(decoded_nxt[ids], collection_nxt[ids])
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
            torch.save(state_prediction.state_dict(), os.path.join(model_dir, 'state_prediction'))
        print('\t\t state prediction:   best epoch: {:d}     \t best metric:{:0.4f}'.format(best_metric[0], best_metric[1]))
        return best_metric

def collect_gt_prob(prob_list, action_nxt, mask_nxt, mortality, gamma=0.99):
    prob_list = [p.data.cpu().numpy() for p in prob_list]

    action_nxt = action_nxt.data.cpu().numpy()
    assert len(action_nxt.shape) == 3 and action_nxt.shape[2] == 3
    action_nxt = action_nxt[:, :, 0] * 49 + action_nxt[:, :, 1] * 7 + action_nxt[:, :, 2] 

    decoded_nxt = np.zeros([action_nxt.shape[0], action_nxt.shape[1], 343])
    for i in range(7):
        for j in range(7):
            for k in range(7):
                decoded_nxt[:, :, i * 49 + j* 7 + k] = prob_list[0][ :, :, i] * prob_list[1][ :, :, j] * prob_list[2][ :, :, k]



    probs_list = []
    rewas_list = []
    preds_list = []
    trues_list = []
    for dn, an, mn, mo in zip(decoded_nxt, action_nxt, mask_nxt.data.cpu().numpy(), mortality.data.cpu().numpy()):
        prob_list = []
        rewa_list = []
        pred_list = []
        true_list = []
        mo = 2 * (0.5 - mo[0])
        for d, a, m in zip(dn, an, mn):
            if m < 0.5:
                rewa_list.append(mo)
                mo = mo * gamma
                prob_list.append(d[int(a)])
                true_list.append(int(a))
                pred_list.append(np.argmax(d))
        rewa_list = rewa_list[::-1]
        probs_list.append(prob_list)
        rewas_list.append(rewa_list)
        preds_list.append(pred_list)
        trues_list.append(true_list)
    return probs_list, rewas_list, preds_list, trues_list

def train_eval_action_imitation(epoch, lr, action_imitation, optimizer, enc_criterion, loader, best_metric, phase):
    if args.phase == 'train':
        action_imitation.train()
    else:
        action_imitation.eval()

    result_dict = dict()


    loss_list = []
    sum_correct, sum_valid = 0.00001, 0
    probs_list, rewas_list, preds_list, trues_list, morts_list = [], [], [], [], []
    for data in tqdm(loader):
        data[:-1] = [Variable(_cuda(x)) for x in data[:-1]]
        collection_crt, collection_nxt, mask_crt, mask_nxt, action_crt, action_nxt, mortality, reward, short_reward, estimated_mortality = data[:10]
        prob_list, next_list = action_imitation(collection_crt, action_crt)
        if phase != 'train':
            ps_list, rs_list, pr_list, ts_list = collect_gt_prob(prob_list, action_nxt, mask_nxt[:, :, 0], mortality)
            probs_list += ps_list
            rewas_list += rs_list
            preds_list += pr_list
            trues_list += ts_list
            # print(len(probs_list), len(rewas_list))

        # loss, n_correct, n_valid = enc_criterion(decoded_nxt, action_nxt, mask_nxt[:, :, 0], reward)
        loss, n_correct, n_valid = short_loss(prob_list, action_nxt, mask_nxt[:, :, 0], short_reward)
        q_value_list, error_list = enc_criterion(prob_list, action_nxt, mask_nxt, reward)

        if phase == 'test':
            for id in data[-1]:
                id = int(id)
                result_dict[id] = dict()
            for i_a in range(3):
                for id, prob, mask in zip(data[-1], prob_list[i_a].data.cpu().numpy(), mask_nxt.data.cpu().numpy()):
                    id = int(id)
                    result_dict[id]['mask'] = mask[:, 0]
                    result_dict[id][str(i_a)] = prob

        # print(n_correct, n_valid)
        sum_correct += n_correct
        sum_valid += n_valid
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            for i_a, q_value, d_error in zip(range(3), q_value_list, error_list):
                # break
                if i_a == 2:
                    retain_graph = False
                else:
                    retain_graph = True
                q_value.backward(0.1 * d_error.data, retain_graph=retain_graph)
            optimizer.step()
        else:
            # print(estimated_mortality.size())
            estimated_mortality = estimated_mortality.data.cpu().numpy()
            # print(estimated_mortality.shape, estimated_mortality.max())
            action_nxt_data = action_nxt.data.cpu().numpy()
            mask_data = mask_crt.data.cpu().numpy()[:, :, 0].reshape(-1)
            mort = []
            bs = len(collection_crt)
            for i in range(3):
                em = estimated_mortality[:, i, :, :].reshape((-1, 7))
                # ac = action_nxt_data[:, :, i].reshape(-1)
                ac = prob_list[i].data.cpu().numpy().argmax(2).reshape(-1)
                assert len(mask_data) == len(em) == len(ac)
                em = em[list(range(len(em))), list(ac)]
                em[mask_data > 0.5] = 0
                for j in range(0, len(em), bs):
                    em_p = em[j: j+bs]
                    em_p = em_p[em_p > 0]
                    if len(em_p) > 0:
                        em_max = em_p.max()
                        em_mean = em_p.mean()
                        mort.append(em_max)
                    else:
                        mort.append(0)
            morts_list.append(mort)
                
        loss_list.append(loss.data.cpu().numpy())
    print('{:s} Epoch {:d} (lr {:0.5f})'.format(phase, epoch, lr))
    print('{:s} loss: {:3.4f} \t'.format(args.loss, np.mean(loss_list))) 
    acc = float(sum_correct) / sum_valid
    print('accuracy: {:3.4f} \t'.format(acc))
    if phase in ['valid', 'test']:
        wis = py_op.compute_wis(probs_list, rewas_list)
        jaccard = py_op.compute_jaccard(preds_list, trues_list)
        em = py_op.compute_estimated_mortality(morts_list)
        print('wis: {:3.4f} \t'.format(wis))
        print('jaccard: {:3.4f} \t'.format(jaccard))
        print('estimated mortality: {:3.4f} \t'.format(em))

        # if acc > best_metric[2]:
        # if jaccard > best_metric[3]:
        if phase == 'valid':
            if wis > best_metric[1]:
                best_metric = [epoch, wis, acc, jaccard, em]
                torch.save(action_imitation.state_dict(), os.path.join(model_dir, 'action_imitation'))
            print('\t\t action imitation:   best epoch: {:d}     wis:{:0.4f}    acc:{:0.4f}    jaccard:{:1.4f}     estimated mortality:{:1.3f}'.format(best_metric[0], best_metric[1], best_metric[2], best_metric[3], best_metric[4]))
        else:
            torch.save(result_dict, '../data/result_dict.ckpt')
        # print(type(probs_list), type(rewas_list))
        return best_metric

def train_autoencoder(train_loader, valid_loader):

    autoencoder = model.autoEncoder(30, 128)
    autoencoder.load_state_dict(torch.load(os.path.join(model_dir, 'autoencoder')))
    enc_criterion = loss.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    
    best_metric = [0, 10e10]
    for epoch in range(1, args.autoencoder_epochs + 1):
        train_eval_ae_epoch(epoch, args.lr, autoencoder, optimizer, enc_criterion, train_loader, best_metric, 'train')
        best_metric = train_eval_ae_epoch(epoch, args.lr, autoencoder, optimizer, enc_criterion, valid_loader, best_metric, 'valid')

def train_state_prediction(train_loader, valid_loader):

    autoencoder = model.autoEncoder(30, 128)
    autoencoder.load_state_dict(torch.load(os.path.join(model_dir, 'autoencoder')))
    state_prediction = model.statePrediction(30, 128)
    enc_criterion = loss.MSELoss()
    optimizer = torch.optim.Adam(state_prediction.parameters(), lr=args.lr)
    
    best_metric = [0, 10e10]
    for epoch in range(1, args.state_prediction_epochs + 1):
        train_eval_ae_epoch(epoch, args.lr, autoencoder, optimizer, enc_criterion, valid_loader, [0, 0], 'test')
        train_eval_state_prediction(epoch, args.lr, autoencoder, state_prediction, optimizer, enc_criterion, train_loader, best_metric, 'train')
        best_metric = train_eval_state_prediction(epoch, args.lr, autoencoder, state_prediction, optimizer, enc_criterion, valid_loader, best_metric, 'valid')

def train_action_imitation(train_loader, valid_loader):

    # autoencoder = model.autoEncoder(30, 128)
    # autoencoder.load_state_dict(torch.load(os.path.join(model_dir, 'autoencoder')))
    action_imitation = model.actionImitation_3(30, 128)
    # qloss = loss.ShortLoss_3()
    qloss = loss.QLoss_3()
    closs  = loss.ClassifyLoss_3()
    optimizer = torch.optim.Adam(action_imitation.parameters(), lr=args.lr)
    
    best_metric = [0, 0, 0, 0, 0]
    for epoch in range(1, args.action_imitation_epochs + 1):
        enc_criterion = qloss
        # enc_criterion = closs
        # if args.loss == 'dqn':
        #     enc_criterion = qloss
        # else:
        #     enc_criterion = closs
        train_eval_action_imitation(epoch, args.lr, action_imitation, optimizer, enc_criterion, train_loader, best_metric, 'train')
        best_metric = train_eval_action_imitation(epoch, args.lr, action_imitation, optimizer, enc_criterion, valid_loader, best_metric, 'valid')

def test_action_imitation(train_loader, valid_loader):

    # autoencoder = model.autoEncoder(30, 128)
    action_imitation = model.actionImitation_3(30, 128)
    action_imitation.load_state_dict(torch.load(os.path.join(model_dir, 'action_imitation')))
    # qloss = loss.ShortLoss_3()
    qloss = loss.QLoss_3()
    closs  = loss.ClassifyLoss_3()
    optimizer = torch.optim.Adam(action_imitation.parameters(), lr=args.lr)
    
    best_metric = [0, 0, 0, 0, 0]
    epoch = 0
    enc_criterion = qloss
    train_eval_action_imitation(epoch, args.lr, action_imitation, optimizer, enc_criterion, valid_loader, best_metric, 'test')

def main():
    icustayid_split_dict = py_op.myreadjson(os.path.join(args.file_dir, 'mechvent.icustayid_split_dict.json'))
    icustayid_train = icustayid_split_dict['icustayid_train']
    icustayid_valid = icustayid_split_dict['icustayid_valid']

    # id risk
    dataset = data_loader.DataBowl(args, icustayid_train, phase='train')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dataset = data_loader.DataBowl(args, icustayid_valid, phase='valid')
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_mortality_prediction(train_loader, valid_loader)

    if args.use_ci:
        dataset = data_loader.CIDataBowl(args, icustayid_train, phase='train')
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    # train_autoencoder(train_loader, valid_loader)
    # train_state_prediction(train_loader, valid_loader)
    
    train_action_imitation(train_loader, valid_loader)
    test_action_imitation(train_loader, valid_loader)



if __name__ == '__main__':
    main()
