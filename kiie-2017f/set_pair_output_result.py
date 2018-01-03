# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:09:12 2017

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
    
    df_pair = None
    with open(WORK_DIR+'/dataset_model/data_subclass_07to11.pickle', 'rb') as f:
        df_pair =  pickle.load(f)
    df_pair.head()
    
    df_output = df({'pair_id': df_pair['pair_id'].values})
    df_output.head()
    
    #data_edge = pd.read_csv(WORK_DIR+'/res_tmp/input_data_edge_vis_07to11.csv')
    data_edge = pd.read_csv(WORK_DIR+'/gephi_input/input_data_edge_07to11.csv')
    pair_true = data_edge[data_edge['weight']==1][['source_lb', 'target_lb']].values
    set_true_pair = set([tuple(pair) for pair in pair_true])
    df_output['true_07to11'] = [1 if pair in set_true_pair else -1 for pair in df_pair['pair_id'].values]
    df_output.head()
    
    #data_edge = pd.read_csv(WORK_DIR+'/res_tmp/input_data_edge_vis_12to16.csv')
    data_edge = pd.read_csv(WORK_DIR+'/gephi_input/input_data_edge_12to16.csv')
    pair_true = data_edge[data_edge['weight']==1][['source_lb', 'target_lb']].values
    set_true_pair = set([tuple(pair) for pair in pair_true])
    df_output['true_12to16'] = [1 if pair in set_true_pair else -1 for pair in df_pair['pair_id'].values]
    df_output.head()

    df_output.to_csv(WORK_DIR+'/dataset_model/output_result_07to11.csv', index=False)
    with open(WORK_DIR+'/dataset_model/output_result_07to11.pickle', 'wb') as f:
        pickle.dump(df_output, f)


    ##################
    df_pair = None
    with open(WORK_DIR+'/dataset_model/data_subclass_12to16.pickle', 'rb') as f:
        df_pair =  pickle.load(f)
    df_pair.head()
    
    df_output = df({'pair_id': df_pair['pair_id'].values})
    df_output.head()
    
    #data_edge = pd.read_csv(WORK_DIR+'/res_tmp/input_data_edge_vis_12to16.csv')
    data_edge = pd.read_csv(WORK_DIR+'/gephi_input/input_data_edge_12to16.csv')
    pair_true = data_edge[data_edge['weight']==1][['source_lb', 'target_lb']].values
    set_true_pair = set([tuple(pair) for pair in pair_true])
    df_output['true_12to16'] = [1 if pair in set_true_pair else -1 for pair in df_pair['pair_id'].values]
    df_output.head()

    df_output.to_csv(WORK_DIR+'/dataset_model/output_result_12to16.csv', index=False)
    with open(WORK_DIR+'/dataset_model/output_result_12to16.pickle', 'wb') as f:
        pickle.dump(df_output, f)
    