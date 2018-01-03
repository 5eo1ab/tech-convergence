# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:23:38 2017

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
    
    data_node = pd.read_csv(WORK_DIR+'/gephi_input/input_data_node.csv')
    dic_node_id = dict(zip(data_node['id'].values, data_node['label'].values))
    data_node.head()
    
    data_edge = pd.read_csv(WORK_DIR+'/gephi_input/input_data_edge_12to16.csv')
    data_edge.head()
    
    ###################
    """output_p1 = None
    with open(WORK_DIR+'/dataset_model/output_result_07to11.pickle', 'rb') as f:
        output_p1 = pickle.load(f)
    dic_pred_p1 = dict(zip(output_p1['pair_id'].values, output_p1['prediction'].values))
    output_p1.head()

    vis_pred, data_edge_p1 = [], data_edge.copy()
    for src, trg in data_edge_p1[['source', 'target']].values:
        decision = dic_pred_p1[(dic_node_id[src], dic_node_id[trg])]
        if decision == -1: decision = 0
        vis_pred.append(decision)
    data_edge_p1['weight'] = vis_pred
    data_edge_p1.head()
    """
    #####################
    output_p2 = None
    with open(WORK_DIR+'/dataset_model/output_result_12to16.pickle', 'rb') as f:
        output_p2 = pickle.load(f)
    dic_pred_p2 = dict(zip(output_p2['pair_id'].values, output_p2['prediction'].values))
    output_p2.head()

    vis_pred, data_edge_p2 = [], data_edge.copy()
    for src, trg in data_edge_p2[['source', 'target']].values:
        decision = dic_pred_p2[(dic_node_id[src], dic_node_id[trg])]
        if decision == -1: decision = 0
        vis_pred.append(decision)
    data_edge_p2['weight'] = vis_pred
    data_edge_p2.head()
    
    #data_edge_p1.to_csv(WORK_DIR+'/gephi_input/input_data_edge_12to16p.csv', index=False)
    data_edge_p2.to_csv(WORK_DIR+'/gephi_input/input_data_edge_pred.csv', index=False)
    