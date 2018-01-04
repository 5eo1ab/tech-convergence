# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:41:46 2017

@author: Hanbin Seo
"""

import os
#import json
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

#import mysql.connector
import pandas as pd
from pandas import DataFrame as df
import numpy as np

if __name__ == '__main__':

    period = "07to11"    #  period="12to16"  
    data_pat = pd.read_csv(WORK_DIR+'/res_tmp/gby_patent_{}.csv'.format(period))
    data_pat.head()
    
    data_citation, path = None, '/'.join(WORK_DIR.split('/')[:-1])
    with open(path+"/df_raw_citation_v201706.pickle", "rb") as f:
        data_citation = pickle.load(f)
    print("Success Load!")
    
    pat_no_list = data_pat['patent_no'].astype(str).values.tolist()
    data_citing = data_citation[data_citation['patent_no'].isin(pat_no_list)]
    data_citing.head()
    
    data_cited = data_citation[data_citation['cited_patent_no'].isin(pat_no_list)]
    tmp = data_cited.groupby('cited_patent_no').count()
    df_tmp = df({'patent_no': tmp.index.values, 'count_cited': tmp[tmp.columns[-1]].values})
    data_pat['patent_no'] = data_pat['patent_no'].astype(str)
    data_pat = pd.merge(data_pat, df_tmp, how='left', on='patent_no')
    data_pat['count_cited'] = data_pat['count_cited'].fillna(value=0)
    data_pat['count_cited'] = data_pat['count_cited'].astype(int)
    data_pat.head()

    data_pat.to_csv(WORK_DIR+'/res_tmp/gby_patent_{}.csv'.format(period), index=False)
    data_cited.to_csv(WORK_DIR+'/data/citation_cited_{}.csv'.format(period), index=False)
    data_citing.to_csv(WORK_DIR+'/data/citation_citing_{}.csv'.format(period), index=False)
    