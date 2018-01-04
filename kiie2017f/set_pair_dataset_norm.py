# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 00:45:48 2017

@author: Hanbin Seo
"""

import os
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import pandas as pd
from pandas import DataFrame as df
import numpy as np

if __name__ == '__main__':
    
    period = '07to11'   # (or) period = '12to16'
    with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'rb') as f:
        df_pair =  pickle.load(f)
    df_pair.head()

    res_li = []
    for col in df_pair.columns[1:]:
        res_li.append(tuple([col]+ df_pair[col].describe().tolist()))
    df_stat = df(res_li, columns=['feature_nm']+df_pair[col].describe().keys().tolist())
    df_stat.head()

    df_pair_norm = df({'pair_id': df_pair['pair_id'].values})
    for col in df_pair.columns[1:]:
        col_min, col_max = df_stat[df_stat['feature_nm']==col][['min', 'max']].values[0]
        if col_min==0 and col_max==0:
            df_pair_norm[col] = df_pair[col].values
        else:
            df_pair_norm[col] = [(x-col_min)/(col_max-col_min) for x in df_pair[col].values]
    df_pair_norm.head()
    #df_pair_norm.describe()

    ##############
    df_stat.to_csv(WORK_DIR+'/dataset_model/feature_statistic_{}.csv'.format(period), index=False)
    df_pair_norm.to_csv(WORK_DIR+'/dataset_model/data_subclass_norm_{}.csv'.format(period), index=False)
    with open(WORK_DIR+'/dataset_model/data_subclass_norm_{}.pickle'.format(period), 'wb') as f:
        pickle.dump(df_pair_norm, f)
