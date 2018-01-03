# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:53:43 2017

@author: Hanbin Seo
"""

import os
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import numpy as np
import pandas as pd
from pandas import DataFrame as df

if __name__ == '__main__':

    period = '07to11'   # (or) period = '12to16'
    relation = 'CC'    # (or) relation = 'BC'
       
    df_pair = None
    with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'rb') as f:
        df_pair = pickle.load(f)
    df_pair.head()

    data_trg = df({'count_{}'.format(relation): df_pair['count_{}'.format(relation)].values})
    data_trg['subclass_i'] = [i for i, j in df_pair['pair_id'].values]
    data_trg['subclass_j'] = [j for i, j in df_pair['pair_id'].values]
    data_trg.head()

    gby_subcls = pd.read_csv(WORK_DIR+'/res_tmp/gby_subclass_{}.csv'.format(period))
    dict_cnt = dict(zip(gby_subcls['subclass'].values, [0]*len(gby_subcls)))
    for subcls in gby_subcls['subclass'].values:
        dict_cnt[subcls] += data_trg[data_trg['subclass_i']==subcls]['count_{}'.format(relation)].values.sum()
        dict_cnt[subcls] += data_trg[data_trg['subclass_j']==subcls]['count_{}'.format(relation)].values.sum()
        
    res_list = []
    for i, array_value in enumerate(data_trg.values):
        c_ij, s_i, s_j = array_value
        c_ij = c_ij/(dict_cnt[s_i] * dict_cnt[s_j])
        res_list.append(c_ij)
    df_pair['norm_{}'.format(relation)] = res_list
    df_pair.head()

    idx = df_pair.columns.values.tolist().index('count_{}'.format(relation))
    col_list = df_pair.columns[:idx+1].values.tolist()
    col_list.append(df_pair.columns[-1])
    col_list += df_pair.columns[idx+1:-1].values.tolist()
    print(col_list)
    
    df_pair = df_pair[col_list]
    df_pair.head()

    with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'wb') as f:
        pickle.dump(df_pair, f)

