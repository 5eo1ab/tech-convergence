# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:07:47 2017

@author: Hanbin Seo
"""

import os
#import json
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from itertools import combinations
from __utility__function import printProgress

if __name__ == '__main__':
    
    period = "07to11" # (or) period = "12to16"
    data_pat = pd.read_csv(WORK_DIR+'/res_tmp/gby_patent_{}.csv'.format(period))
    data_pat.head()

    df_target = data_pat[data_pat['count_subclass']>1][['patent_no', 'concat_subclass', 'count_cited']]
    df_target.head()

    df_pair = None
    with open(WORK_DIR+'/dataset_model/data_pair_subclass_{}.pickle'.format(period), 'rb') as f:
        df_pair = pickle.load(f)
    df_pair.head()

    df_pair['count_cited'], size_loop = [0]*len(df_pair), len(df_target)
    for i, array_value in enumerate(df_target.values):
        list_comb = list(combinations(array_value[1].split(), 2))
        idx_rep = df_pair[df_pair['pair_id'].isin(list_comb)].index.values
        df_pair.loc[idx_rep, 'count_cited'] += array_value[-1]
        printProgress(i, size_loop)
    df_pair.head()

    df_pair.to_csv(WORK_DIR+"/dataset_model/data_subclass_{}.csv".format(period), index=False)
    with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'wb') as f:
        pickle.dump(df_pair, f)
    print("Success!")
    
    