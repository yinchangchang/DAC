#!/usr/bin/env python
# coding=utf-8


import sys

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import random
import json
from collections import OrderedDict
from tqdm import tqdm
from sklearn import mixture
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB



sys.path.append('../tools')
import parse, py_op

args = parse.args

def generate_action(df):
    setting_th_dict = {
            'TidalVolume': [2.50, 5.00, 7.50, 10.00, 12.50, 15.00],
            'PEEP': [5, 7, 9, 11,13, 15],
            'FiO2_1': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
            }
    setting_list =sorted(setting_th_dict.keys())
    action = np.zeros((len(df), 3))
    mechvent = df['mechvent'].values
    for i_s, s in enumerate(setting_list):
        values = df[s]
        for i_th, th in enumerate(setting_th_dict[s]):
            action[values > th, i_s] = i_th + 1
    print(mechvent.mean(), mechvent.max())
    print(np.mean(mechvent==1))
    print('action', action.max())
    print('action', action.mean(0))
    action[mechvent<1, :] = 0
    # action = action[:, 0] * 49 + action[:, 1] * 7 + action[:, 2]
    # return action
    return action




def generate_data():
    df_origin = pd.read_csv('../data/mechvent_cohort.csv')
    df = df_origin.copy()
    df['mechvent'] = (df['TidalVolume'] + df['PEEP'] + df['FiO2_1']) > 0
    action = generate_action(df)

    binary_fields = ['gender','mechvent','re_admission']
    norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C',
                'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
                'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
                'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index',
                'PaO2_FiO2','cumulated_balance', 'elixhauser', 'Albumin', u'CO2_mEqL', 'Ionised_Ca']
    log_fields = ['max_dose_vaso','SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR',
                'input_total','input_4hourly','output_total','output_4hourly', 'bloc']
    observ_fields = ['gender', 'age','elixhauser','re_admission', 'SOFA', 'SIRS', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2',
                'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium',
                'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 'Albumin', 'Hb', 'WBC_count', 'Platelets_count', 'PTT',
                'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'Arterial_lactate', 'HCO3', 'PaO2_FiO2', 'output_total', 'output_4hourly']


    norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C',
                'Potassium','Sodium','Chloride','Glucose','Magnesium',
                'Hb','WBC_count','Platelets_count','Arterial_pH','paO2',
                'Arterial_BE','HCO3','Arterial_lactate',
                'PaO2_FiO2','Albumin']
    log_fields = ['max_dose_vaso','SpO2','Creatinine','Total_bili',
                'input_total','input_4hourly','output_total','output_4hourly', 'bloc']
    observ_fields = ['gender', 'age','re_admission', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2',
                'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Creatinine', 'Magnesium', 
                'Total_bili', 'Albumin', 'Hb', 'WBC_count', 'Platelets_count',
                'Arterial_pH', 'paO2', 'Arterial_BE', 'Arterial_lactate', 'HCO3', 'PaO2_FiO2', 'output_total', 'output_4hourly']


    df[binary_fields] = df[binary_fields] - 0.5 
    for item in norm_fields:
        av = df[item].mean()
        std = df[item].std()
        df[item] = (df[item] - av) / std
    df[log_fields] = np.log(0.1 + df[log_fields])
    for item in log_fields:
        av = df[item].mean()
        std = df[item].std()
        df[item] = (df[item] - av) / std
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.keys())

    assert len(scaled_df) == len(action)
    scaled_df['action_0'] = action[:, 0]
    scaled_df['action_1'] = action[:, 1]
    scaled_df['action_2'] = action[:, 2]
    print(scaled_df['action_0'].values.shape)
    mortality = df['mortality_90d'].values

    print('action', action.min(), action.max())
    print('mortality', mortality.min(), mortality.max())

    x_list = []
    t_list = []
    y_list = []

    for icustayid in tqdm(scaled_df['icustayid'].unique()):
        ids = scaled_df['icustayid'] == icustayid
        collection_data = scaled_df[ids][observ_fields].values 
        action_data = action[ids, :]
        # mortality_90d = scaled_df[ids]['mortality_90d'].values[-1:]
        mortality_24h = scaled_df[ids]['died_within_48h_of_out_time'].values
        mortality_24h[:-6] = 0

        for i in range(len(action_data) - 1):
            x = collection_data[i]
            t = action_data[i+1]
            y = mortality_24h[i+1:i+2]
            x_list.append(x)
            t_list.append(t)
            y_list.append(y)

        # the action after last record is none
        x_list.append(x)
        t_list.append(t)
        y_list.append([-1])

    # x_list = np.array(x_list)
    # t_list = np.array(t_list)
    # y_list = np.array(y_list)
    data = np.concatenate((np.array(x_list), np.array(t_list), np.array(y_list)), 1)
    print(np.array(t_list).max(), np.array(t_list).min())
    print(data.shape)
    data[np.isnan(data)] = 0
    np.save('../data/mechevent_data.npy', data)
    return data

def treatment_prediction(X, Y, sample_weight=None):
    clf = GaussianNB()
    clf.fit(X, Y, sample_weight=sample_weight)
    proba = clf.predict_proba(X)
    # print('Mean output', clf.predict(X).mean())
    # print('Mean output', clf.predict(X).mean(), '    mean label', Y.mean())
    return proba


def mortality_prediction(X,Y, sample_weight=None):
    # clf = GaussianNB()
    clf = BernoulliNB()
    clf.fit(X, Y, sample_weight=sample_weight)
    proba = clf.predict_proba(X)
    return clf, proba[:, 1]

def gmm_estimation():
    # generate_data()
    data_init = np.load('../data/mechevent_data.npy')
    mortality_90d  = data_init[:, -1]
    data = data_init[mortality_90d>=0, :]
    # print('data', data_init.shape, data.shape)
    # return

    # compute p(t|x)
    proba_list = []
    for i in range(3):
        collection_data = data[:, :-4]
        action_data = data[:, -4 + i].astype(np.int32)
        # p = np.identity(7)
        # print(action_data.min(), action_data.max())
        # action_data = p[action_data]
        # p = bayesian_prediction(collection_data, action_data)
        p = treatment_prediction(collection_data, action_data)
        proba_list.append(p)
    
    # compute p(y|x,t)
    # resample treatments
    sample_p_list = []
    for i in tqdm(range(len(data))):
        action = data[i, -4:-1]
        p = 1
        for a, proba in zip(action, proba_list):
            p *= proba[i, int(a)]
        sample_p_list.append(p)
    sample_p_list = np.array(sample_p_list)
    sample_p_list[sample_p_list < 0.001] = 0.001
    weight = 1 / sample_p_list

    xt = data[:, :-1]
    y = data[:, -1]
    # p_y_t  = bayesian_prediction(xt, y, sample_weight=weight)[:, 1]
    clf_reweight, p_y_t  = mortality_prediction(xt, y, sample_weight=weight)

    x = data[:, :-4]
    y = data[:, -1]
    clf_init, p_y_x  = mortality_prediction(x, y)


    print('Mean Treatment probability {:1.3f} of {:d} records of survival patients'.format( np.mean(sample_p_list[y<0.5]), len(sample_p_list[y<0.5])))
    print('Mean Treatment probability {:1.3f} of {:d} records of mortality patients'.format(np.mean(sample_p_list[y>0.5]), len(sample_p_list[y>0.5])))

    # compute reward
    p_y_t = clf_reweight.predict_proba(data_init[:, :-1])[:, 1]
    p_y_x = clf_init.predict_proba(data_init[:, :-4])[:, 1]
    reward = p_y_x - p_y_t
    std = reward.std()
    mean = reward.mean()
    reward = reward - mean
    mean_pos_reward = reward[reward > 0].mean()
    # reward[reward>0] = mean_pos_reward + reward[reward>0] 
    # reward[(p_y_t - p_y_t.mean() < 0) * (reward < 0)] = mean_pos_reward
    # reward[(reward < 0) * (reward > - reward.std())] = 0
    reward[reward > 0] += mean_pos_reward
    reward[(reward < 0) * (p_y_t < p_y_t.mean() + p_y_t.std() )] = mean_pos_reward
    reward[(reward < 0) * (reward > - std)] = 0
    reward_init = reward
    reward = reward_init[mortality_90d>=0]

    print('Mean reward {:1.3f} of {:d} treatment records of survival patients; positive reward rate: {:1.4f}'.format( np.mean(reward[y<0.5]), len(reward[y<0.5]), np.mean(reward[y<0.5] > 0)))
    print('Mean reward {:1.3f} of {:d} treatment records of mortality patients; positive reward rate: {:1.4f}'.format(np.mean(reward[y>0.5]), len(reward[y>0.5]), np.mean(reward[y>0.5] > 0)))
    print('Mean reward {:1.3f} of {:d} treatment records of survival patients; negative reward rate: {:1.4f}'.format( np.mean(reward[y<0.5]), len(reward[y<0.5]), np.mean(reward[y<0.5] < 0)))
    print('Mean reward {:1.3f} of {:d} treatment records of mortality patients; negative reward rate: {:1.4f}'.format(np.mean(reward[y>0.5]), len(reward[y>0.5]), np.mean(reward[y>0.5] < 0)))


    df = pd.read_csv('../data/mechvent_cohort_reward.csv')
    df['p_y_x'] = p_y_x 
    df['p_y_t'] = p_y_t
    df['reward'] = reward_init
    df['is_last'] = mortality_90d < 0
    df.to_csv('../data/mechvent_cohort_reward.csv')




def adjust_reward():
    df = pd.read_csv('../data/mechvent_cohort_reward.csv')
    p_y_t = df['p_y_t'].values
    p_y_x = df['p_y_x'].values
    is_last = df['is_last'].values
    mechvent = df['mechvent']
    mortality_90d = df['mortality_90d']
    print(p_y_t.shape)
    p_y_t = p_y_t[is_last == False]
    p_y_x = p_y_x[is_last == False]
    mechvent = mechvent[is_last == False]
    mortality_90d = mortality_90d[is_last == False]
    print(p_y_t.shape)
    p_y_t = p_y_t[mechvent == 1]
    p_y_x = p_y_x[mechvent == 1]
    mortality_90d = mortality_90d[mechvent == 1]
    y = mortality_90d
    print(p_y_t.shape)

    reward = p_y_x - p_y_t
    reward = reward - reward.mean()
    mean_pos_reward = reward[reward > 0].mean()
    # reward[reward > - 0.5 * reward.std()] = mean_pos_reward
    reward[p_y_t - p_y_t.mean() < p_y_t.std()] = mean_pos_reward
    std = reward

    print('Mean reward {:1.3f} of {:d} treatment records of survival patients; positive reward rate: {:1.4f}'.format( np.mean(reward[y<0.5]), len(reward[y<0.5]), np.mean(reward[y<0.5] >= 0)))
    print('Mean reward {:1.3f} of {:d} treatment records of mortality patients; positive reward rate: {:1.4f}'.format(np.mean(reward[y>0.5]), len(reward[y>0.5]), np.mean(reward[y>0.5] >= 0)))
    

def Bayesion_estimation(i_a):
    print('------------------------------------------------------------')
    print(i_a)
    data_init = np.load('../data/mechevent_data.npy')
    mortality_90d  = data_init[:, -1]
    data = data_init[mortality_90d>=0, :]
    # print('data', data_init.shape, data.shape)
    # return

    # compute p(t|x)
    collection_data = data[:, :-4]
    action_data = data[:, -4 + i_a : -4 + i_a + 1].astype(np.int32)
    proba = treatment_prediction(collection_data, action_data)
    
    # compute p(y|x,t)
    # resample treatments
    sample_p_list = []
    # for i in tqdm(range(len(data))):
    for i in range(len(data)):
        action = data[i, -4 + i_a]
        p = proba[i, int(action)]
        sample_p_list.append(p)
    sample_p_list = np.array(sample_p_list)
    print(sample_p_list.mean(), sample_p_list.std())
    print('>0.6', np.mean(sample_p_list > 0.6))
    print('<0.1', np.mean(sample_p_list < 0.1))
    sample_p_list[sample_p_list < 0.02] = 0.02
    weight = 1 / sample_p_list

    # ids = list(range(data_init.shape[1] - 4)) + [-4 + i_a]
    t = data[:, -4 + i_a].astype(np.int32)
    y = data[:, -1]
    p_y_t_proba = np.zeros(7)
    for i_t in range(7):
        p = np.mean(y[t == i_t])
        # y_selected = y[t == i_t]
        # w_selected = weight[t == i_t]
        # p = np.dot(y_selected, w_selected) / np.sum(w_selected)
        p_y_t_proba[i_t] = p 
    p_y_t = p_y_t_proba[t]
    print('p_y_t', p_y_t.shape)
    print('p_y_t_proba', p_y_t_proba.shape)

    x = data[:, :-4]
    y = data[:, -1]
    clf_init, p_y_x  = mortality_prediction(x, y)


    print('Mean Treatment probability {:1.3f} of {:d} records of survival patients'.format( np.mean(sample_p_list[y<0.5]), len(sample_p_list[y<0.5])))
    print('Mean Treatment probability {:1.3f} of {:d} records of mortality patients'.format(np.mean(sample_p_list[y>0.5]), len(sample_p_list[y>0.5])))

    # compute reward
    p_y_x = clf_init.predict_proba(data_init[:, :-4])[:, 1]

    t = data_init[:, -4 + i_a].astype(np.int32)
    p_y_t = p_y_t_proba[t]

    reward = p_y_x - p_y_t
    std = reward.std()
    mean = reward.mean()
    reward = reward - mean
    mean_pos_reward = reward[reward > 0].mean()
    # reward[reward>0] = mean_pos_reward + reward[reward>0] 
    # reward[(p_y_t - p_y_t.mean() < 0) * (reward < 0)] = mean_pos_reward
    # reward[(reward < 0) * (reward > - reward.std())] = 0
    reward[reward > 0] += mean_pos_reward
    reward[(reward < 0) * (p_y_t < p_y_t.mean() + p_y_t.std() )] = mean_pos_reward
    reward[(reward < 0) * (reward > - std)] = 0
    reward_init = reward
    reward = reward_init[mortality_90d>=0]

    print('Mean reward {:1.3f} of {:d} treatment records of survival patients; positive reward rate: {:1.4f}'.format( np.mean(reward[y<0.5]), len(reward[y<0.5]), np.mean(reward[y<0.5] > 0)))
    print('Mean reward {:1.3f} of {:d} treatment records of mortality patients; positive reward rate: {:1.4f}'.format(np.mean(reward[y>0.5]), len(reward[y>0.5]), np.mean(reward[y>0.5] > 0)))
    print('Mean reward {:1.3f} of {:d} treatment records of survival patients; negative reward rate: {:1.4f}'.format( np.mean(reward[y<0.5]), len(reward[y<0.5]), np.mean(reward[y<0.5] < 0)))
    print('Mean reward {:1.3f} of {:d} treatment records of mortality patients; negative reward rate: {:1.4f}'.format(np.mean(reward[y>0.5]), len(reward[y>0.5]), np.mean(reward[y>0.5] < 0)))


    if i_a == 0:
        df = pd.read_csv('../data/mechvent_cohort.csv')
    else:
        df = pd.read_csv('../data/mechvent_cohort_reward.csv')

    df['p_y_x'] = p_y_x 
    df['action_{:d}'.format(i_a)] = data_init[:, -4 + i_a ].astype(np.int32)
    df['reward_' + str(i_a)] = reward_init
    df['is_last'] = (mortality_90d < 0).astype(np.int32)

    for a in range(7):
        df['p_y_t_{:d}_{:d}'.format(i_a, a)] = p_y_t_proba[a]
        print(a, p_y_t_proba[a])
    df.to_csv('../data/mechvent_cohort_reward.csv')






def main():
    # df_origin = pd.read_csv('../data/mechvent_cohort.csv')
    # df = df_origin.copy()
    # print(df['died_within_48h_of_out_time'].values.mean())
    # print(df['died_in_hosp'].values.mean())
    # return

    generate_data()
    # Bayesion_estimation(2)
    # return
    for i in range(3):
        Bayesion_estimation(i)
    # gmm_estimation()
    # adjust_reward()





if __name__ == '__main__':
    main()
