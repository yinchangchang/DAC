#!/usr/bin/env python

import os
import json
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('../tools')
import parse, py_op

mortality_prediction_window = 6

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
    action[mechvent<1, :] = 0
    action_3 = action
    action = action[:, 0] * 49 + action[:, 1] * 7 + action[:, 2]
    return action, action_3



class DataBowl(Dataset):
    def __init__(self, args, icustayid_list, phase='train', dataset='mimic'):
        assert (phase == 'train' or phase == 'valid' or phase == 'test')
        self.dataset = dataset
        self.args = args
        self.phase = phase
        self.icustayid_list = icustayid_list

        self.action_risk = torch.load('../file/{:s}.icustayid_risk_dict.ckpt'.format('mechvent'))

        # df_origin = pd.read_csv(os.path.join(args.data_dir, dataset+'table.csv'))
        df_origin = pd.read_csv(os.path.join(args.data_dir, 'mechvent_cohort_reward.csv'))
        df = df_origin.copy()

        # action = generate_action(df)
        # df['mechvent'] = (df['TidalVolume'] + df['PEEP'] + df['FiO2_1']) > 0


        binary_fields = ['gender','mechvent','re_admission']
        norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
                'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
                'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
                'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index',
                'PaO2_FiO2','cumulated_balance', 'elixhauser', 'Albumin', u'CO2_mEqL', 'Ionised_Ca']
        log_fields = ['max_dose_vaso','SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR',
                'input_total','input_4hourly','output_total','output_4hourly', 'bloc']
        self.observ_fields = ['gender', 'age','elixhauser','re_admission', 'SOFA', 'SIRS', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2',
                'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium',
                'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 'Albumin', 'Hb', 'WBC_count', 'Platelets_count', 'PTT',
                'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'Arterial_lactate', 'HCO3', 'PaO2_FiO2', 'output_total', 'output_4hourly']

        _norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C',
                'Potassium','Sodium','Chloride','Glucose','Magnesium',
                'Hb','WBC_count','Platelets_count','Arterial_pH','paO2',
                'Arterial_BE','HCO3','Arterial_lactate',
                'PaO2_FiO2','Albumin']
        _log_fields = ['max_dose_vaso','SpO2','Creatinine','Total_bili',
                'input_total','input_4hourly','output_total','output_4hourly', 'bloc']
        self._observ_fields = ['gender', 'age','re_admission', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2',
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
        scaled_df['died_in_hosp'] = df_origin['died_in_hosp']
        scaled_df['died_within_48h_of_out_time'] = df_origin['died_within_48h_of_out_time']
        scaled_df['mortality_90d'] = df_origin['mortality_90d']
        scaled_df['icustayid'] = df_origin['icustayid']
        scaled_df['action_0'] = df_origin['action_0']
        scaled_df['action_1'] = df_origin['action_1']
        scaled_df['action_2'] = df_origin['action_2']
        # assert len(scaled_df) == len(action)
        # scaled_df['action'] = action
        # action = scaled_df['action'].values
        # print('action', action.shape, action.mean())
        self.scaled_df = scaled_df
        

    def augment(self, data):
        return data


    def __getitem__(self, idx, split=None):
        icustayid = self.icustayid_list[idx]
        n = 50
        collection_data = self.scaled_df[self.scaled_df['icustayid'] == icustayid][self.observ_fields].values # [k, 25]
        action_data = [self.scaled_df[self.scaled_df['icustayid'] == icustayid]['action_' + str(i)].values  for i in range(3)]
        action_data = np.array(action_data).transpose(1, 0)
        short_reward = [self.scaled_df[self.scaled_df['icustayid'] == icustayid]['reward_' + str(i)].values  for i in range(3)]
        short_reward = np.array(short_reward).transpose(1, 0)
        # short_reward += 0.05
        short_reward[short_reward < - 0.05] = - 0.05
        # short_reward[short_reward >=  short_reward.min()] = 1
        # print(action_data.shape, action_data)
        mortality_90d = self.scaled_df[self.scaled_df['icustayid'] == icustayid]['mortality_90d'].values[-1:].astype(np.float32)
        mortality_48h = self.scaled_df[self.scaled_df['icustayid'] == icustayid]['died_within_48h_of_out_time'].values.astype(np.float32)
        # mortality_48h = self.scaled_df[self.scaled_df['icustayid'] == icustayid]['mortality_90d'].values.astype(np.float32)
        assert len(mortality_48h) > 1
        size = collection_data.shape
        if size[0] > mortality_prediction_window and mortality_48h[-1] > 0.5:
            mortality_48h[:- mortality_prediction_window] = 0

        delta = 1
        if size[0] >= n + delta:
            collection_data = collection_data[- n - delta:]
            action_data = action_data[ -n - delta:]
            short_reward = short_reward[ -n - delta:]
            mask = np.zeros_like(collection_data)
            mortality_48h = mortality_48h[- n - delta:]
            reward = np.zeros(n, dtype=np.float32)
            reward[ - 1] = (0.5 - mortality_90d) * 2
        else:
            padding = np.zeros((n + delta - size[0], size[1]))
            collection_data = np.concatenate((collection_data, padding), 0)
            mortality_48h = np.concatenate((mortality_48h, padding[:,0]), 0)
            action_data = np.concatenate((action_data, padding[:, :3]), 0)
            short_reward = np.concatenate((short_reward, padding[:, :3]), 0)
            mask = np.zeros_like(collection_data)
            mask[size[0]:, :] = 1
            reward = np.zeros(n, dtype=np.float32)
            # print(len(reward), mortality_90d)
            reward[size[0] - 1] = (0.5 - mortality_90d) * 2

        mortality_48h = mortality_48h[:- delta].astype(np.float32)
        collection_data = collection_data.astype(np.float32)
        mask = mask.astype(np.float32)
        action_data = action_data.astype(np.int64)
        collection_crt = collection_data[: - delta]
        collection_nxt = collection_data[delta :]
        mask_crt = mask[: - delta]
        mask_nxt = mask[delta :]
        action_crt = action_data[: - delta]
        action_nxt = action_data[delta :]
        short_reward = short_reward.astype(np.float32)[: - delta]
        try:
            estimated_mortality = np.array(self.action_risk[str(int(icustayid))])
        except:
            estimated_mortality = np.array([0])
        # risk = np.array([self.id_risk[icustayid]], dtype=np.float32)
        # print('action', action_crt.min(), action_crt.max(), action_crt.shape)

        collection_crt[np.isnan(collection_crt)] = 0

        return torch.from_numpy(collection_crt), torch.from_numpy(collection_nxt), \
                torch.from_numpy(mask_crt), torch.from_numpy(mask_nxt), \
                torch.from_numpy(action_crt), torch.from_numpy(action_nxt), \
                torch.from_numpy(mortality_90d), torch.from_numpy(reward), \
                torch.from_numpy(short_reward), torch.from_numpy(estimated_mortality), icustayid

    def __len__(self):
        return len(self.icustayid_list)

class CIDataBowl(Dataset):
    def __init__(self, args, icustayid_list, phase='train', dataset='mimic'):
        assert (phase == 'train' or phase == 'valid' or phase == 'test')
        self.args = args
        self.phase = phase
        self.icustayid_list = icustayid_list
        assert dataset == 'mimic'

        self.action_risk = torch.load('../file/{:s}.icustayid_risk_dict.ckpt'.format(dataset))
        df_origin = pd.read_csv(os.path.join(args.data_dir, dataset+'table.csv'))
        print('---------------------------')
        print(len(icustayid_list))
        print(len(self.action_risk))
        print('---------------------------')

        if phase == 'train':
            icustayid_risk_dict = json.load(open('../file/icustayid_risk_dict.json'))
            case_list = []
            control_list = []
            id_risk = { }
            for id in icustayid_list:
                risk = icustayid_risk_dict[str(int(id))][1]
                id_risk[id] = risk
                if risk > 0:
                    case_list.append(id)
                else:
                    control_list.append(id)
            if args.use_ci in [1, 3]:
                self.icustayid_list = case_list
            self.case_list = case_list
            self.case_set = set(case_list)
            self.control_list = control_list
            self.id_risk = id_risk
        df = df_origin.copy()

        # df_action = df[['icustayid', 'charttime', 'median_dose_vaso', 'input_4hourly']].copy()
        vaso_data = np.array(df['max_dose_vaso'])
        # print('vaso_data', vaso_data.shape)
        vaso_dose_list = [0.45, 0.22, 0.08, 0]
        vaso_action = np.zeros_like(vaso_data) + 4
        for i in range(len(vaso_dose_list)):
            dose = vaso_dose_list[i]
            vaso_action[vaso_data <= dose] = 3 - i
            # df_action['vaso'][df_action['median_dose_vaso'] <=  dose] = 3 - i
        # print('vaso mean', vaso_action.mean(), '  median', sorted(vaso_action)[ int(len(vaso_action )/ 2)])
        # for i in range(5):
        #     print('vaso', i, np.mean(vaso_action==i))
        iv_data = np.array(df['input_4hourly'])
        iv_dose_list = [530, 180, 50, 0]
        # df_action['iv'] = 4
        iv_action = np.zeros_like(iv_data) + 4
        for i in range(len(iv_dose_list)):
            dose = iv_dose_list[i]
            # df_action['iv'][df_action['input_4hourly'] <=  dose] = 3 - i
            iv_action[iv_data <= dose] = 3 - i
        # for i in range(5):
        #     print('fluid', i, np.mean(iv_action==i))
        # print('iv mean', iv_action.mean(), '  median', sorted(iv_action)[ int(len(iv_action)/ 2)])
        # df_action['action'] = df_action['vaso'] * 5 + df_action['iv']
        # self.df_action = df_action
        action = vaso_action * 5 + iv_action
        # action = vaso_action 


        binary_fields = ['gender','mechvent','re_admission']
        norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
                'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
                'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
                'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index',
                'PaO2_FiO2','cumulated_balance', 'elixhauser', 'Albumin', u'CO2_mEqL', 'Ionised_Ca']
        log_fields = ['max_dose_vaso','SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR',
                'input_total','input_4hourly','output_total','output_4hourly', 'bloc']
        self.observ_fields = ['gender', 'age','elixhauser','re_admission', 'SOFA', 'SIRS', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2',
                'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium',
                'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 'Albumin', 'Hb', 'WBC_count', 'Platelets_count', 'PTT',
                'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'Arterial_lactate', 'HCO3', 'PaO2_FiO2', 'output_total', 'output_4hourly']

        _binary_fields = ['gender','re_admission']
        _norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
                'Potassium','Sodium','Glucose','Magnesium','Hb','WBC_count','Platelets_count','Arterial_pH','paO2','paCO2',
                'Arterial_BE','HCO3','Arterial_lactate','PaO2_FiO2','Albumin']
        _log_fields = ['max_dose_vaso','SpO2','Creatinine','Total_bili','input_4hourly','output_4hourly']
        self._observ_fields = ['gender', 'age','re_admission', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2',
                'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Glucose', 'Creatinine', 'Magnesium', 'Total_bili', 'Albumin', 'Hb', 
                'WBC_count', 'Platelets_count', 'Arterial_pH', 'paO2', 'Arterial_BE', 'Arterial_lactate', 'HCO3', 'PaO2_FiO2', 
                'output_4hourly']

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
        scaled_df['died_in_hosp'] = df_origin['died_in_hosp']
        scaled_df['died_within_48h_of_out_time'] = df_origin['died_within_48h_of_out_time']
        scaled_df['mortality_90d'] = df_origin['mortality_90d']
        scaled_df['icustayid'] = df_origin['icustayid']
        assert len(scaled_df) == len(action)
        scaled_df['action'] = action
        action = scaled_df['action'].values
        # print('action', action.shape, action.mean())
        self.scaled_df = scaled_df
        

    def augment(self, data):
        return data

    def get_paired_patient(self, icustayid):
            risk = self.id_risk[icustayid]
            if icustayid in self.case_set:
                control_list = self.control_list
            else:
                control_list = self.case_list
            control_pool = []
            for cont in control_list:
                delta = abs(self.id_risk[cont] - risk)
                control_pool.append([delta, cont])
            control_pool = sorted(control_pool, key=lambda s:s[0])[:100]
            np.random.shuffle(control_pool)
            return control_pool[0][1]

    def __getitem__(self, idx, split=None):
        icustayid = self.icustayid_list[idx]
        if self.phase == 'train':
            data = []
            case_data = self.__getitem__icustdayid(icustayid)
            data.append(case_data)
            paired_id = self.get_paired_patient(icustayid)
            # print(paired_id)
            control_data = self.__getitem__icustdayid(paired_id)
            data.append(control_data)
            if self.args.use_ci == 3:
                control_pool = self.control_list[:]
                np.random.shuffle(control_pool)
                icustayid = control_pool[0]
                case_data = self.__getitem__icustdayid(icustayid)
                data.append(case_data)
                paired_id = self.get_paired_patient(icustayid)
                control_data = self.__getitem__icustdayid(paired_id)
                data.append(control_data)

            new_data = []
            for i in range(len(data[0]) - 1):
                item = np.array([x[i] for x in data])
                new_data.append(torch.from_numpy(item))
            new_data.append([x[-1] for x in data])
            return new_data
        else:
            data = self.__getitem__icustdayid(icustayid)
            data[:-1] = [torch.from_numpy(x) for x in data[:-1]]
            return data

    def __getitem__icustdayid(self, icustayid):
        n = 30
        collection_data = self.scaled_df[self.scaled_df['icustayid'] == icustayid][self.observ_fields].values # [k, 25]
        action_data = self.scaled_df[self.scaled_df['icustayid'] == icustayid]['action'].values # k
        mortality_90d = self.scaled_df[self.scaled_df['icustayid'] == icustayid]['mortality_90d'].values[-1:].astype(np.float32)
        mortality_48h = self.scaled_df[self.scaled_df['icustayid'] == icustayid]['died_within_48h_of_out_time'].values.astype(np.float32)
        # mortality_48h = self.scaled_df[self.scaled_df['icustayid'] == icustayid]['mortality_90d'].values.astype(np.float32)
        size = collection_data.shape

        if size[0] > mortality_prediction_window and mortality_48h[-1] > 0.5:
            mortality_48h[:- mortality_prediction_window] = 0

        delta = 1
        if size[0] >= n + delta:
            collection_data = collection_data[- n - delta:]
            action_data = action_data[ -n - delta:]
            mortality_48h = mortality_48h[- n - delta:]
            mask = np.zeros_like(collection_data)
        else:
            padding = np.zeros((n + delta - size[0], size[1]))
            collection_data = np.concatenate((collection_data, padding), 0)
            mortality_48h = np.concatenate((mortality_48h, padding[:,0]), 0)
            action_data = np.concatenate((action_data, padding[:, 0]), 0)
            mask = np.zeros_like(collection_data)
            mask[size[0]:, :] = 1
            reward = np.zeros(n, dtype=np.float32)
            # reward[size[0] - 1] = 
            r = (0.5 - mortality_90d) * 2
            i = size[0] - 1
            while i>= 0:
                reward[i] = r
                r *= 0.99
                i -= 1


        mortality_48h = mortality_48h[:- delta].astype(np.float32)
        collection_data = collection_data.astype(np.float32)
        mask = mask.astype(np.float32)
        action_data = action_data.astype(np.int64)
        collection_crt = collection_data[: - delta]
        collection_nxt = collection_data[delta :]
        mask_crt = mask[: - delta]
        mask_nxt = mask[delta :]
        action_crt = action_data[: - delta]
        action_nxt = action_data[delta :]
        risk = np.array([self.id_risk[icustayid]], dtype=np.float32)

        estimated_mortality = self.action_risk[str(int(icustayid))]
        assert len(estimated_mortality.shape) == 2
        if len(estimated_mortality) > n:
            estimated_mortality = estimated_mortality[:n]
        else:
            padding = np.zeros((n-len(estimated_mortality), 25), dtype=np.float32)
            estimated_mortality = np.concatenate((estimated_mortality, padding), 0)

        return collection_crt, collection_nxt, \
                mask_crt, mask_nxt, \
                action_crt, action_nxt, \
                mortality_90d, reward, estimated_mortality, mortality_48h, icustayid

        return torch.from_numpy(collection_crt), torch.from_numpy(collection_nxt), \
                torch.from_numpy(mask_crt), torch.from_numpy(mask_nxt), \
                torch.from_numpy(action_crt), torch.from_numpy(action_nxt), \
                torch.from_numpy(mortality_90d), torch.from_numpy(reward), icustayid

    def __len__(self):
        return len(self.icustayid_list)
