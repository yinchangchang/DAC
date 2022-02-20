#!/usr/bin/env python
# coding=utf-8


import sys

import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.append('../tools')
import parse, py_op

args = parse.args

def split_mimic():
    data = pd.read_csv(os.path.join(args.data_dir, 'mechvent_cohort_reward.csv'))
    icustayid_set = set(data['icustayid'])
    print('There are {:d} icu stays.'.format(len(icustayid_set)))
    icustayid_list = list(icustayid_set)
    np.random.shuffle(icustayid_list)
    n_train = int(0.6 * len(icustayid_list))
    n_valid = int(0.2 * len(icustayid_list))
    icustayid_train = icustayid_list[: n_train]
    icustayid_valid = icustayid_list[n_train : n_train + n_valid]
    icustayid_test  = icustayid_list[n_train + n_valid :]
    icustayid_split_dict = {
            'icustayid_train': icustayid_train,
            'icustayid_valid': icustayid_valid,
            'icustayid_test': icustayid_test,
            }
    py_op.mywritejson(os.path.join(args.file_dir, 'mechvent.icustayid_split_dict.json'), icustayid_split_dict)


def main():
    split_mimic()
    # split_ast()

if __name__ == '__main__':
    main()
