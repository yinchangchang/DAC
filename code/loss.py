#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import *

import sys
sys.path.append('../tools')
import parse, py_op

args = parse.args


class ClassifyLoss(nn.Module):
    def __init__(self):
        super(ClassifyLoss, self).__init__()

    def forward(self, output, labels, mask, reward=None):
        # print('label', labels.size())

        assert len(output.size()) == 3

        # if reward is not None and args.loss == 'dqn':
        # if 0:
        if args.phase == 'train':
            assert len(reward.size()) == 2
            # reward = reward.max(1)[0]
            # ids = reward > 0.5
            reward = reward.min(1)[0]
            ids = reward < -0.5
            assert len(reward.size()) == 1
            assert len(reward) == len(labels) == len(mask) == len(output)

            output = output[ids]
            mask = mask[ids]
            labels = labels[ids]
            # print(mask.size())

        output = output.view(-1, output.size(2))
        labels = labels.view(-1)
        mask = mask.view(-1)

        # print(output.size(), labels.size(), mask.size())
        assert len(output) == len(labels) == len(mask)
        assert output.size(1) == 343

        ids = mask < 0.5
        # print(output.size())
        # print(labels.size())
        # print(ids.size())
        output = output[ids, :]
        labels = labels[ids]

        if len(labels):
            loss = F.nll_loss(torch.log(output), labels)
            output = output.data.cpu().numpy().argmax(1)
            labels = labels.data.cpu().numpy()
            n_valid = len(labels)

            idx = labels > 0
            labels = labels[idx]
            output = output[idx]
            
            n_correct = (output== labels).sum()
            n_valid = len(labels)
            # if n_correct == 0:
            #     print(n_correct, n_valid, _valid)
            return loss, n_correct, n_valid
        else:
            return 0, 0, 0


class ClassifyLoss_3(nn.Module):
    def __init__(self):
        super(ClassifyLoss_3, self).__init__()

    def forward(self, output_list, labels_3, mask, reward=None):
        # print('label', labels_3.size())
        # print('mask', mask.size())

        assert len(output_list) == 3
        mask = mask.view(-1)
        loss = 0
        correct = 1
        for i in range(3):
            output = output_list[i]
            labels = labels_3[:, :, i]

            output = output.view(-1, output.size(2))
            labels = labels.view(-1)

            assert len(output) == len(labels) == len(mask)
            assert output.size(1) == 7

            ids = mask < 0.5
            output = output[ids, :]
            labels = labels[ids]
            assert len(labels) > 0

            loss += F.nll_loss(torch.log(output), labels)

            pred = output.data.cpu().numpy().argmax(1)
            labels = labels.data.cpu().numpy()

            correct *= (pred == labels).astype(np.int32)
        n_correct = correct.sum()
        n_valid = len(correct)

        return loss, n_correct, n_valid



class ShortLoss_3(nn.Module):
    def __init__(self):
        super(ShortLoss_3, self).__init__()

    def forward(self, output_list, labels_3, mask, reward):
        # print('label', labels_3.size())
        # print('mask', mask.size())

        assert len(output_list) == 3
        mask = mask.view(-1)
        loss = 0
        correct = 1
        for i in range(3):
            output = output_list[i]
            labels = labels_3[:, :, i]

            output = output.view(-1, output.size(2))
            labels = labels.view(-1)
            sr = reward[:, :, i].view(-1)

            assert len(output) == len(labels) == len(mask) == len(sr)
            assert output.size(1) == 7

            ids = mask < 0.5
            output = output[ids, :]
            labels = labels[ids]
            sr = sr[ids]
            assert len(labels) > 0

            # loss += F.nll_loss(torch.log(output), labels)
            op = output[range(len(output)), list(labels.data.cpu().numpy())]
            # print('output', output.size(), op.size())
            # sr = Variable(torch.from_numpy(np.ones_like(sr.data.cpu().numpy())).cuda())
            loss -= torch.dot(torch.log(op), sr) / len(sr)
            # print(loss)

            pred = output.data.cpu().numpy().argmax(1)
            labels = labels.data.cpu().numpy()

            correct *= (pred == labels).astype(np.int32)
        n_correct = correct.sum()
        n_valid = len(correct)

        return loss, n_correct, n_valid


class QLoss_3(nn.Module):
    def __init__(self):
        super(QLoss_3, self).__init__()
        self.gamma = 0.99

    def forward(self, output_list, labels_3, mask, reward):
        # print('label', labels_3.size())
        # print('mask', mask.size())

        assert len(output_list) == 3
        q_value_list, error_list = [], []
        for i in range(3):
            output = output_list[i]
            q_value = torch.log(output)
            action_nxt = labels_3[:, :, i]
            # print('q_value', q_value.size())
            # print('action_nxt', action_nxt.size())

            q_value = q_value * (1 - mask[:, :, :7])
            q_size = list(q_value.size())
            q_value = q_value.view(-1, q_size[2])
            action_nxt = action_nxt.view(-1)
            # print('q_size', q_size)
            current_q_value = q_value[list(range(len(q_value))), list(action_nxt.data.cpu().numpy().astype(np.int32))].view(q_size[:2])
            q_value = q_value.view(q_size)
            action_nxt = action_nxt.view([q_size[0], -1])
            next_q_value = q_value.detach().max(2)[0]

            # update_q_value = current_q_value[:, :-1]

            current_q_value = current_q_value[:, :-1]
            current_reward = reward[:, 1:]
            next_q_value = next_q_value[:, 1:]

            target_q_value = current_reward + (self.gamma * next_q_value)
            bellman_error = target_q_value - current_q_value
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            d_error = clipped_bellman_error * -1.0

            q_value_list.append(current_q_value)
            error_list.append(d_error)

        return q_value_list, error_list


