# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 00:46:24 2017

@author: Hanbin Seo
"""

import os
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import pandas as pd
from pandas import DataFrame as df
from itertools import combinations
from math import factorial as fact

from set_subclass_Mfr import print_count_subcls_list
from __utility__function import printProgress

def dict_add_count(dict_, key_):
    dict_[key_] += 1
    return None

"""def calculate_nC2(num):
    return num*(num-1)//2
"""

if __name__ == '__main__':
    
    period = "07to11"   # (or) period = "12to16"
    data_read = pd.read_csv(WORK_DIR+'/res_tmp/pair_pat_subcls_{}.csv'.format(period))
    data_read.head()
      
    gby_pat = pd.read_csv(WORK_DIR+'/res_tmp/gby_patent_{}.csv'.format(period))
    gby_pat.head()
    
    gby_subcls = pd.read_csv(WORK_DIR+'/res_tmp/gby_subclass_{}.csv'.format(period))
    gby_subcls.head()
    
    comb_li  = list(combinations(gby_subcls['subclass'].values, 2))
    dict_subcls_comb_count = dict(zip( comb_li, [0]*len(comb_li)))
    for i, array_value in enumerate(gby_pat.values):
        if array_value[1] == 1:
            continue
        elif array_value[1] == 2:
            dict_add_count(dict_subcls_comb_count, tuple(array_value[-1].split()))
        else:
            for comb in list(combinations(array_value[-1].split(), 2)):
                dict_add_count(dict_subcls_comb_count, comb)
        printProgress(i, len(gby_pat))
    df_pair = df(list(dict_subcls_comb_count.items()), columns=['pair_id', 'count_cooccur'])
    df_pair.head() 
 
    df_pair.to_csv(WORK_DIR+'/dataset_model/data_subclass_{}.csv'.format(period), index=False)    
    with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'wb') as f:
        pickle.dump(df_pair, f)
    print("Success!")
    

    """
    ################################
    ## Test for total count of occur of each subclass
    ################################
    dict_subcls_test = {}
    list_pat_no = gby_pat['patent_no'].values.tolist()    
    for i, pat_no in enumerate(list_pat_no):
        arr_subcls = data_read[data_read['patent_no']==pat_no]['subclass'].values
        if len(arr_subcls) == 1:
            dict_add_count(dict_subcls_test, arr_subcls[-1])
        else:
            num_occur = calculate_nC2(len(arr_subcls))
            for subcls in arr_subcls:
                dict_add_count(dict_subcls_test, subcls)
        print("{}/{}".format(i+1, len(list_pat_no)))

    gby_subcls['count_test'] = [dict_subcls_test[subcls] for subcls in gby_subcls['subclass'].values]
    gby_subcls['is_true'] = gby_subcls['patent_no'].values-gby_subcls['count_test'].values
    print(len(gby_subcls[gby_subcls['is_true']!=0]))
    """
    
    
    
    