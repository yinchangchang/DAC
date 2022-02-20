# -*- coding: utf-8 -*-
import os
import json
import traceback
from collections import OrderedDict 
import random
import numpy as np
# from fuzzywuzzy import fuzz

import sys

################################################################################
### pre define variables
#:: enumerate
#:: raw_input
#:: listdir
#:: sorted
### pre define function
def mywritejson(save_path,content):
    content = json.dumps(content,indent=4,ensure_ascii=False)
    with open(save_path,'w') as f:
        f.write(content)

def myreadjson(load_path):
    with open(load_path,'r') as f:
        return json.loads(f.read())

def mywritefile(save_path,content):
    with open(save_path,'w') as f:
        f.write(content)

def myreadfile(load_path):
    with open(load_path,'r') as f:
        return f.read()

def myprint(content):
    print(json.dumps(content,indent=4,ensure_ascii=False)) 

def rm(fi):
    os.system('rm ' + fi)

def mystrip(s):
    return ''.join(s.split())

def mysorteddict(d,key = lambda s:s, reverse=False):
    dordered = OrderedDict()
    for k in sorted(d.keys(),key = key,reverse=reverse):
        dordered[k] = d[k]
    return dordered

def mysorteddictfile(src,obj):
    mywritejson(obj,mysorteddict(myreadjson(src)))

from multiprocessing import Pool, Manager
manager = Manager()
matchDict = manager.dict()
def match(src, objs, grade):
        for obj in objs:
            value = fuzz.partial_ratio(src,obj)
            if value > grade:
                matchDict[src] = matchDict.get(src, []) + [[value, obj]]
                if len(matchDict[src]) > 1:
                    print('------------------------------') 
        print(len(matchDict)) 

def myfuzzymatch(srcs,objs,grade=80):
    p = Pool(32)
    for src in list(srcs): # [:100]:
        p.apply_async(match, args=(src, objs, grade))
    p.close()
    p.join()

    new_match = { k:sorted(v, reverse=True) for k,v in matchDict.items() }

    return new_match

def mydumps(x):
    return json.dumps(content,indent=4,ensure_ascii=False)

def get_random_list(l,num=-1,isunique=0):
    if isunique:
        l = set(l)
    if num < 0:
        num = len(l)
    if isunique and num > len(l):
        return 
    lnew = []
    l = list(l)
    while(num>len(lnew)):
        x = l[int(random.random()*len(l))]
        if isunique and x in lnew:
            continue
        lnew.append(x)
    return lnew

def fuzz_list(node1_list,node2_list,score_baseline=66,proposal_num=10,string_map=None):
    node_dict = { }
    for i,node1 in enumerate(node1_list):
        match_score_dict = { }
        for node2 in node2_list:
            if node1 != node2:
                if string_map is not None:
                    n1 = string_map(node1)
                    n2 = string_map(node2)
                    score = fuzz.partial_ratio(n1,n2)
                    if n1 == n2:
                        node2_list.remove(node2)
                else:
                    score = fuzz.partial_ratio(node1,node2)
                if score > score_baseline:
                    match_score_dict[node2] = score
            else:
                node2_list.remove(node2)
        node2_sort = sorted(match_score_dict.keys(), key=lambda k:match_score_dict[k],reverse=True)
        node_dict[node1] = [[n,match_score_dict[n]] for n in node2_sort[:proposal_num]]
        print(i,len(node1_list)) 
    return node_dict, node2_list

def swap(a,b):
	return b, a

def mkdir(data_dir):
    if not os.path.exists(data_dir):
        try:
            os.mkdir(data_dir)
        except:
            mkdir('/'.join(data_dir.split('/')[:-1]))
            os.mkdir(data_dir)

def compute_wis(probs_list, rs_list, gamma=0.99):
    # print(type(probs_list), len(probs_list))
    # print(type(rs_list), len(rs_list))
    pho_list = []
    for probs, rs in zip(probs_list, rs_list):
        # print(len(probs), len(rs))
        assert len(probs) == len(rs)
        pho = []
        for prob, r in zip(probs, rs):
            # assert prob >= 0
            # assert prob <= 1
            prob = min(1, max(0, prob))
            if len(pho) == 0:
                pho.append(prob)
            else:
                pho.append(prob * pho[-1])
        pho_list.append(pho)
    max_step = max([len(pho) for pho in pho_list])
    w_list = []
    for i in range(max_step):
        w_h = []
        for pho in pho_list:
            if len(pho) > i:
                w_h.append(pho[i])
            # elif len(pho):
            #     w_h.append(pho[len(pho)-1])
        w_list.append(np.mean(w_h))

    v_list = []
    for pho, rs in zip(pho_list, rs_list):
        h = len(pho)
        if h <= 1:
            continue
        assert pho[h-1] <= 1
        assert w_list[h-1] <= 1
        assert len(rs) > 0
        assert rs[-1] != 0
        v_wis = pho[h-1] / w_list[h-1] * rs[-1] * np.power(gamma, len(rs))
        v_list.append(v_wis)
    return np.mean(v_list)

def compute_jaccard(preds_list, trues_list):
    sims = []
    for pred_list, true_list in zip(preds_list, trues_list):
        assert len(pred_list) == len(true_list)
        sim = []
        n = 6
        for i in range(0, len(pred_list), n):
            ps = pred_list[i:i+n]
            ts = true_list[i:i+n]
            ps = [49 + int(a / 49)  for a in ps] + [a % 7  for a in ps] + [7 + int((a % 49)/7)  for a in ps]
            ts = [49 + int(a / 49)  for a in ts] + [a % 7  for a in ts] + [7 + int((a % 49)/7)  for a in ps]
            cap = len(set(ps) & set(ts))
            cup = len(set(ps) | set(ts))
            sim.append(float(cap)/cup)
        if len(sim):
            sims.append(np.mean(sim))
    return np.mean(sims)

def compute_estimated_mortality(morts_list):
    ems = []
    for mt in morts_list:
        if len(mt):
            # ems.append(np.max(mt))
            ems.append(np.mean(mt))
    return np.mean(ems)

