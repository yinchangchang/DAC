import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from time import time
import pickle as pkl
import pandas as pd
from collections import Counter
from sklearn.metrics import r2_score
# from tensorboardX import SummaryWriter
# % matplotlib inline



class autoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(autoEncoder, self).__init__()
        
        self.encoder = nn.LSTM(input_size, hidden_size, dropout=0.2, batch_first=True)
        # self.action_embedding = nn.Embedding (25, hidden_size)
        self.fc_crt = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, input_size),
        )
        self.fc_nxt = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, input_size),
        )

    
    def forward(self, x):

        # action_embedding = self.action_embedding(action)
        # x = torch.cat((x, action_embedding), 2)
        # print(x.size())

        # print('input', x.size())
        hidden_state , _ = self.encoder(x)
        # print('hidden', hidden_state.size())
        output_crt = self.fc_crt(hidden_state)
        output_nxt = self.fc_nxt(hidden_state)
        # print('output', output_crt.size(), output_nxt.size())
        return hidden_state, output_crt, output_nxt


class statePrediction(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(statePrediction, self).__init__()
        
        self.action_embedding = nn.Embedding (25, hidden_size)
        self.input_size = input_size
        self.encoder = nn.LSTM(input_size, hidden_size, dropout=0.2, batch_first=True)
        self.fc_nxt = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 25 * input_size),
        )
    
    def forward(self, x):
        hidden_state_nxt, _ = self.encoder(x)
        output_nxt = self.fc_nxt(hidden_state_nxt)
        size = list(output_nxt.size())
        return output_nxt.view([size[0], size[1], 25, self.input_size])


class _actionImitation(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(actionImitation, self).__init__()
        
        self.action_embedding = nn.Embedding (25, hidden_size)
        # self.fc = nn.Linear ( hidden_size + 30, 25)
        self.fc = nn.Linear ( 30, 25)
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, hidden_state, action, adapt=0):
        # print('action', action.size())
        # action_embedding = self.action_embedding(action)
        # print('action embedding', action_embedding.size())

        # action_embedding = self.action_embedding(action)
        # print('embedding', action_embedding.size())
        output_nxt = self.fc(hidden_state)
        # torch.cat((hidden_state, action_embedding), 2)
        # print('output', output_nxt.size())
        prob = self.softmax(output_nxt)
        if adapt:
            return output_nxt, prob, hidden_state_nxt.detach()
        else:
            return output_nxt, prob

class actionImitation(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(actionImitation, self).__init__()
        
        self.action_embedding = nn.Embedding (343, hidden_size)
        # self.encoder = nn.LSTM(hidden_size, hidden_size, dropout=0.2, batch_first=True)
        self.encoder = nn.LSTM(45, hidden_size, dropout=0.2, batch_first=True)
        self.fc_nxt = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 343),
        )
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, hidden_state, action=None, adapt=0):
        # print('action', action.size())
        # action_embedding = self.action_embedding(action)
        # print('action embedding', action_embedding.size())

        # action_embedding = self.action_embedding(action)
        # hidden_state_nxt, _ = self.encoder(torch.cat((hidden_state, action_embedding), 2))
        hidden_state_nxt, _ = self.encoder(hidden_state)
        output_nxt = self.fc_nxt(hidden_state_nxt)
        prob = self.softmax(output_nxt)
        if adapt:
            return output_nxt, prob, hidden_state_nxt.detach()
        else:
            return output_nxt, prob


class mortalityPrediction(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(mortalityPrediction, self).__init__()
        
        self.action_embedding = nn.Embedding (25, hidden_size)
        self.encoder = nn.LSTM(input_size + hidden_size, hidden_size, dropout=0.2, batch_first=True)
        self.fc_nxt = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 25),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden_state, action):
        action_embedding = self.action_embedding(action)
        hidden_state_nxt, _ = self.encoder(torch.cat((hidden_state, action_embedding), 2))
        output_nxt = self.fc_nxt(hidden_state_nxt).transpose(2, 1)
        return self.sigmoid(self.maxpool(output_nxt)).view(-1)


class estimatePrediction(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(estimatePrediction, self).__init__()
        
        self.action_embedding = nn.Embedding (25, hidden_size)
        self.encoder = nn.LSTM(input_size + hidden_size, hidden_size, dropout=0.2, batch_first=True)
        self.fc_nxt = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            # nn.Linear ( hidden_size, 25),
            nn.Linear ( hidden_size, 1),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden_state, action):
        action_embedding = self.action_embedding(action)
        hidden_state_nxt, _ = self.encoder(torch.cat((hidden_state, action_embedding), 2))
        output_nxt = self.fc_nxt(hidden_state_nxt)
        return self.sigmoid(output_nxt)

class Critic_s(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(Critic_s, self).__init__()
        
        self.encoder = nn.LSTM(hidden_size, hidden_size, dropout=0.2, batch_first=True)
        self.fc_nxt = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 1),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden_state):
        hidden_state_nxt, _ = self.encoder(hidden_state)
        output_nxt = self.fc_nxt(hidden_state_nxt)
        return self.sigmoid(output_nxt)

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(Critic, self).__init__()
        
        self.encoder = nn.LSTM(hidden_size, hidden_size, dropout=0.2, batch_first=True)
        self.fc_nxt = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 25),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden_state):
        hidden_state_nxt, _ = self.encoder(hidden_state)
        output_nxt = self.fc_nxt(hidden_state_nxt)
        return self.sigmoid(output_nxt)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(Actor, self).__init__()
        
        self.action_embedding = nn.Embedding (25, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, dropout=0.2, batch_first=True)
        self.fc_nxt = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 25),
        )
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, hidden_state):
        hidden_state_nxt, _ = self.encoder(hidden_state)
        output_nxt = self.fc_nxt(hidden_state_nxt)
        prob = self.softmax(output_nxt)
        return output_nxt, prob


class actionImitation_3(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(actionImitation_3, self).__init__()
        
        # self.encoder = nn.LSTM(hidden_size, hidden_size, dropout=0.2, batch_first=True)
        self.encoder = nn.LSTM(45, hidden_size, dropout=0.2, batch_first=True)
        self.fc_1 = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 7),
        )
        self.fc_2 = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 7),
        )
        self.fc_3 = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 7),
        )
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, hidden_state, action=None, adapt=0):
        hidden_state_nxt, _ = self.encoder(hidden_state)
        next_1 = self.fc_1(hidden_state_nxt)
        next_2 = self.fc_2(hidden_state_nxt)
        next_3 = self.fc_3(hidden_state_nxt)
        prob_1 = self.softmax(next_1)
        prob_2 = self.softmax(next_2)
        prob_3 = self.softmax(next_3)
        return [prob_1, prob_2, prob_3], [next_1, next_2, next_3]

class estimatePrediction_3(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(estimatePrediction_3, self).__init__()
        
        # self.action_embedding = nn.Embedding (25, hidden_size)
        self.encoder = nn.LSTM(input_size, hidden_size, dropout=0.2, batch_first=True)
        self.fc_1 = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 7),
        )
        self.fc_2 = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 7),
        )
        self.fc_3 = nn.Sequential (
            nn.Linear ( hidden_size, hidden_size),
            nn.ReLU ( ),
            nn.Linear ( hidden_size, 7),
        )
        # self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden_state, action):
        # action_embedding = self.action_embedding(action)
        # hidden_state_nxt, _ = self.encoder(torch.cat((hidden_state, action_embedding), 2))
        # output_nxt = self.fc_nxt(hidden_state_nxt)
        hidden_state_nxt, _ = self.encoder(hidden_state)
        prob_1 = self.sigmoid(self.fc_1(hidden_state_nxt))
        prob_2 = self.sigmoid(self.fc_2(hidden_state_nxt))
        prob_3 = self.sigmoid(self.fc_3(hidden_state_nxt))
        return [prob_1, prob_2, prob_3]

