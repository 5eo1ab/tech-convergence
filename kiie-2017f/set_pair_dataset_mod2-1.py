# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:53:43 2017

@author: Hanbin Seo
"""

import os
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import pandas as pd
from itertools import combinations

def get_comb_subclass(pat_pair, dict_concat):
    subcls_list = dict_concat[pat_pair[0]].split()
    subcls_list += dict_concat[pat_pair[-1]].split()
    subcls_list = list(set(subcls_list))
    res_list = combinations(subcls_list, 2)
    return list(res_list)

if __name__ == '__main__':

    period = '07to11'   # (or) period = '12to16'
    key = 'count_BC'    # (or) key = 'count_CC'
    
    arg_tmp = input("period Is 07to11 or 12to16 ? (07to11=1, 12to16=2) ")
    if arg_tmp == '2': period = '12to16'
    arg_tmp = input("relation Is CC else BC? (CC=1, BC=2) ")
    if arg_tmp == '1': key = 'count_CC'
    
    df_pair = None
    with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'rb') as f:
        df_pair = pickle.load(f)
    df_pair.head()
    
    data_pat = pd.read_csv(WORK_DIR+'/res_tmp/gby_patent_{}.csv'.format(period))
    dict_concat = dict(zip(data_pat['patent_no'].values, data_pat['concat_subclass'].values))
    data_trg = pd.read_csv(WORK_DIR+'/res_tmp/relation_{}_pat_{}.csv'.format(key.split('_')[-1], period))
    dict_cnt = dict(zip(df_pair['pair_id'].values, [0]*len(df_pair)))
    print("Read {} @ {}".format(key.split('_')[-1], period))

    for i, array_value in enumerate(data_trg[data_trg.columns[:-1]].values): 
        #list_comb = list(combinations(get_subclass_list(array_value[:-1], dict_concat), 2))
        list_comb = get_comb_subclass(array_value[:-1], dict_concat)
        for pair in list_comb:  
            if pair in dict_cnt.keys(): dict_cnt[pair] += array_value[-1]
        print("Run {} @ {}\t{}/{}".format(key.split('_')[-1], period, i+1, len(data_trg)))
    df_pair[key] = [dict_cnt[pair] for pair in df_pair['pair_id'].values]   

    df_pair.to_csv(WORK_DIR+"/dataset_model/data_subclass_{}.csv".format(period), index=False)
    with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'wb') as f:
        pickle.dump(df_pair, f)
    print("Success! {}".format(period))
    
    