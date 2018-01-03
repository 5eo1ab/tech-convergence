# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:29:15 2017

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

from set_subclass_Mfr import print_count_subcls_list

def get_labeled_node_df(data_node, target_set):
    data_node['id'] = ["n{}".format(i) for i in range(len(data_node))]
    data_node['label'] = data_node['subclass'].values
    res_df = data_node[['id', 'label', 'count_07to11', 'count_12to16']]
    res_df['tech_type'] = ['Svc.' if lb in target_set else 'Mfr.{}'.format(lb[0]) for lb in res_df['label'].values]
    #res_df['tech_type'] = [target_lb if lb in target_set else else_lb for lb in res_df['label'].values]
    #res_df['tech_sector'] = [lb[0] for lb in res_df['label'].values]
    return res_df
def get_tuple_dict2edge_df(pair_df, node_df, type_='undirected', is_raw_count=False):
    res_df = df({'count': pair_df['count_cooccur'].values})
    res_df['source_lb'], res_df['target_lb'] = zip(*pair_df['pair_id'])
    
    dict_subcls_id = dict(zip(node_df['label'], node_df['id']))
    res_df['source'] = [dict_subcls_id[subcls] for subcls in res_df['source_lb'].values]
    res_df['target'] = [dict_subcls_id[subcls] for subcls in res_df['target_lb'].values]
    
    if is_raw_count:
        res_df['weight'] = res_df['count'].values
    else:
        dict_subcls_cnt = dict(zip(node_df['label'], node_df['count']))
        arr_src = [dict_subcls_cnt[subcls] for subcls in res_df['source_lb'].values]
        arr_trg = [dict_subcls_cnt[subcls] for subcls in res_df['target_lb'].values]
        arr_src, arr_trg, arr_coo = np.array(arr_src), np.array(arr_trg), res_df['count'].values
        #res_df['weight'] = (arr_coo*len(node_df))/(arr_src*arr_trg)  # proximity index
        res_df['weight'] = (arr_coo*(10**6))/(arr_src*arr_trg)           # association strength * 1000
    print("weight (min, max) = ({}, {})".format(res_df['weight'].min(), res_df['weight'].max()))
    
    res_df['type'] = [type_]*len(res_df)
    return res_df[['source', 'target', 'source_lb', 'target_lb', 'count', 'weight', 'type']]


if __name__ == '__main__':
       
    period_str_list = ["07to11", "12to16"]
    
    ## input format of node data
    node_07to11 = pd.read_csv(WORK_DIR+'/res_tmp/gby_subclass_07to11.csv')
    node_07to11.columns = ['subclass', 'count_07to11']
    node_07to11.head()
    
    node_12to16 = pd.read_csv(WORK_DIR+'/res_tmp/gby_subclass_12to16.csv')
    node_12to16.columns = ['subclass', 'count_12to16']
    node_12to16.head()    

    data_node = pd.merge(node_07to11, node_12to16, on='subclass')    
    data_node.head()
    
    with open(WORK_DIR+'/data_json/subclass_list_res07to16.json') as f:
        dict_subcls_list = json.load(f)
    print_count_subcls_list(dict_subcls_list)
    print("len @ data_node: {}\nlen @ subclass of Tol.: {}".format(len(data_node), len(dict_subcls_list['subclass_Tot.'])))
    
    set_subcls_Svc = set(dict_subcls_list['subclass_Svc.'])
    node_df = get_labeled_node_df(data_node, set_subcls_Svc)
    #node_df = get_labeled_node_df(data_node, set_subcls_Svc, 'Svc.', 'Mfr.')
    node_df.head()

    edge_list = []
    node_df.to_csv(WORK_DIR+'/gephi_input/input_data_node.csv', index=False)
    for period in period_str_list:
        with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'rb') as f:
            data_pair = pickle.load(f)
        edge_df = get_tuple_dict2edge_df(data_pair, node_df, is_raw_count=True)
        #edge_df.to_csv(WORK_DIR+'/gephi_input/input_data_edge_{}.csv'.format(period), index=False)
        edge_list.append(edge_df)
    print("Sccuess!")        

    size, cutoff = len(edge_list[0]), 250
    for edge_df in edge_list:
        edge_df['weight'] = [1 if cnt>cutoff else 0 for cnt in edge_df['count'].values]
        print(len(edge_df[edge_df['count']>cutoff]), end=', ')
        print(len(edge_df[edge_df['count']>cutoff])/size)
    """ cutoff = 100  
        243, 0.0680672268907563
        680, 0.19047619047619047
        
        cutoff = 50
        380, 0.10644257703081232
        967, 0.27086834733893556
        
        cutoff = 150
        167, 0.04677871148459384
        543, 0.15210084033613444

        cutoff = 200
        133, 0.03725490196078431
        443, 0.12408963585434174

        cutoff = 250
        110, 0.03081232492997199
        381, 0.10672268907563025

        cutoff = 300
        97, 0.027170868347338936
        323, 0.09047619047619047

        cutoff = 400
        73, 0.020448179271708684
        260, 0.07282913165266107

        cutoff = 500
        58, 0.016246498599439777
        208, 0.05826330532212885
    """

    for i, period in enumerate(period_str_list):
        edge_list[i].to_csv(WORK_DIR+'/gephi_input/input_data_edge_{}.csv'.format(period), index=False)

