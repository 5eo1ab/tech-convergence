# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 00:58:52 2017

@author: Hanbin Seo
"""

import os
import json
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import pandas as pd
from pandas import DataFrame as df

from set_subclass_Mfr import print_count_subcls_list

if __name__ == '__main__':

    period = "07to11"  # (or)  period = "12to16" 
    period_raw = "2007to2011" if period is "07to11" else "2012to2016"    
    data_raw = pd.read_csv(WORK_DIR+"/data/data_{}.csv".format(period_raw))
    data_raw.head()
    
    data = data_raw[['patent_no', 'subclass']]
    print(data.shape)   # (1194478, 2)
    data.head()
    
    dict_subcls_list = json.load(open(WORK_DIR+'/data_json/subclass_list_res07to16.json'.format(period)))
    print_count_subcls_list(dict_subcls_list)
    
    data_res = data[data['subclass'].isin(dict_subcls_list['subclass_Tot.'])]
    print(data_res.shape) # (1194050, 2)
    data_res.head()
    
    
    tmp = data_res.groupby('patent_no').count()
    df_tmp = df({'patent_no': tmp.index.values,
                 'count_subclass': tmp[tmp.columns[-1]].values})
    df_tmp.head()
    tmp =  data_res.groupby('patent_no')['subclass'].apply(lambda x: " ".join(x.astype(str)))
    df_tmp['concat_subclass'] = tmp.values
    df_tmp.tail(15)

    gby_res_pat = df_tmp[['patent_no', 'count_subclass', 'concat_subclass']].copy()
    gby_res_pat.head()

    tmp = data_res.groupby('subclass').count()
    df_tmp = df({'subclass': tmp.index.values, 
                 'count_patent': tmp[tmp.columns[-1]].values})
    df_tmp.head() 
    
    gby_res_class = df_tmp[['subclass', 'count_patent']].copy()
    gby_res_class.head()
    
    ############# 
    data_res.to_csv(WORK_DIR+"/res_tmp/pair_pat_subcls_{}.csv".format(period), index=False)
    gby_res_pat.to_csv(WORK_DIR+'/res_tmp/gby_patent_{}.csv'.format(period), index=False)
    gby_res_class.to_csv(WORK_DIR+'/res_tmp/gby_subclass_{}.csv'.format(period), index=False)
    
