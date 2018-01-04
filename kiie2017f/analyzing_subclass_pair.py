# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:14:06 2017

@author: Hanbin Seo
"""

import os
import json
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import pandas as pd
from pandas import DataFrame as df
import numpy as np


if __name__ == '__main__':
    
    dict_subcls_list = None
    with open(WORK_DIR+'/data_json/subclass_list_res07to16.json') as f:
        dict_subcls_list = json.load(f)
    dict_subcls_list.keys()
    svc_list = dict_subcls_list['subclass_Svc.']
    
    subcls_desc = None
    with open(WORK_DIR+"/data_json/subclass_description.json") as f:
        subcls_desc = json.load(f)
    
    
    ###################
    
    df_output = None
    with open(WORK_DIR+'/dataset_model/output_result_07to11.pickle', 'rb') as f:
        df_output = pickle.load(f)
    df_output.head()

    df_output['new_connect'] = df_output['true_12to16'].values - df_output['true_07to11'].values
    print("Count of positive: {}".format(len(df_output[df_output['new_connect']>0]['pair_id'])))
    print("Count of negative: {}".format(len(df_output[df_output['new_connect']<0]['pair_id'])))
    df_output.head()
    
    df_pos = df(df_output[df_output['new_connect']>0]['pair_id'].values.tolist())
    df_pos_res = df_pos[df_pos[0].isin(svc_list)&~df_pos[1].isin(svc_list)]
    df_pos_res = df_pos_res.append(df_pos[df_pos[1].isin(svc_list)&~df_pos[0].isin(svc_list)])
    print(len(df_pos_res)) # 84

    df_res = df(np.array([pair if pair[0] in svc_list else pair[::-1] for pair in df_pos_res.values]),
                columns = [['Svc_subclass', 'Mfr_subclass']])
    df_res.head()

    for key in df_res.columns:
        new_key = "{}_description".format(key.split('_')[0])
        df_res[new_key] = [subcls_desc[subcls] for subcls in df_res[key].values]
    df_res.head()

    df_res.to_csv(WORK_DIR+'/gephi_output/new_pair_in_12to16.csv', index=False)

    
    df_neg = df(df_output[df_output['new_connect']<0]['pair_id'].values.tolist())
    df_neg_res = df_neg[df_neg[0].isin(svc_list)&~df_neg[1].isin(svc_list)]
    df_neg_res = df_neg_res.append(df_neg[df_pos[1].isin(svc_list)&~df_neg[0].isin(svc_list)])
    print(len(df_neg_res)) # 0
    df_neg_res.head()

    ################
    df_output = None
    with open(WORK_DIR+'/dataset_model/output_result_12to16.pickle', 'rb') as f:
        df_output = pickle.load(f)
    df_output.head()

    df_output['new_connect'] = df_output['prediction']-df_output['true_12to16']
    print("Count of positive: {}".format(len(df_output[df_output['new_connect']>0]['pair_id'])))
    print("Count of negative: {}".format(len(df_output[df_output['new_connect']<0]['pair_id'])))
    df_output.head()

    #df_output[df_output['new_connect']<0]

    df_pos = df(df_output[df_output['new_connect']>0]['pair_id'].values.tolist())
    df_pos_res = df_pos[df_pos[0].isin(svc_list)&~df_pos[1].isin(svc_list)]
    df_pos_res = df_pos_res.append(df_pos[df_pos[1].isin(svc_list)&~df_pos[0].isin(svc_list)])
    print(len(df_pos_res)) # 132
    df_pos_res.head()
    
    df_res = df(np.array([pair if pair[0] in svc_list else pair[::-1] for pair in df_pos_res.values]),
                columns = [['Svc_subclass', 'Mfr_subclass']])
    df_res.head()

    for key in df_res.columns:
        new_key = "{}_description".format(key.split('_')[0])
        df_res[new_key] = [subcls_desc[subcls] for subcls in df_res[key].values]
    df_res.head()

    df_res.to_csv(WORK_DIR+'/gephi_output/new_pair_in_predict.csv', index=False)

    df_res.groupby('Svc_subclass').count()['Mfr_subclass']
    """ A01K     3
        B65B     3
        G01G     2
        G01R    30
        G06F     9
        G06G    11
        G06Q    30
        G07B     5
        G07C    12
        G07F     8
        H04M    19 
    """    
    
    df_neg = df(df_output[df_output['new_connect']<0]['pair_id'].values.tolist())
    df_neg_res = df_neg[df_neg[0].isin(svc_list)&~df_neg[1].isin(svc_list)]
    df_neg_res = df_neg_res.append(df_neg[df_pos[1].isin(svc_list)&~df_neg[0].isin(svc_list)])
    print(len(df_neg_res)) # 21
    df_neg_res.head()
    
    
    