# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 21:49:23 2017

@author: Hanbin Seo
"""

import os
import json
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import pandas as pd
from pandas import DataFrame as df

def print_count_subcls_list(dict_subcls_list):
    for k, v in dict_subcls_list.items():
        if v is not None: print("Count of {}:\t{}".format(k, len(v)))
        else: print("Count of {}:\t{}".format(k, 0))
    return None

if __name__ == '__main__':
    
    data_raw1 = pd.read_csv(WORK_DIR+"/data/data_2007to2011.csv")
    data_raw2 = pd.read_csv(WORK_DIR+"/data/data_2012to2016.csv")
    data_raw = data_raw1.append(data_raw2)
    print("{} + {} = {}, {}".format(len(data_raw1), len(data_raw2), len(data_raw1)+len(data_raw2), len(data_raw)))
    data_raw.head()
    
    data = data_raw[['patent_no', 'subclass']]
    print("Count of patent: {}".format(len(data.groupby('subclass').count())))
    print("Count of patent: {}".format(len(data.groupby('patent_no').count())))
    data.head()
    # Count of subclass: 1555
    # Count of patent: 927223
    
    dict_subcls_list = json.load(open(WORK_DIR+'/data_json/subclass_list_ini.json'))
    print(dict_subcls_list.keys())
    print_count_subcls_list(dict_subcls_list)
    
    tmp = data[data['subclass'].isin(dict_subcls_list['subclass_Svc.'])]['patent_no'].values
    pat_no_list = list(set(tmp))
    data_pat_Svc = data[data['patent_no'].isin(pat_no_list)]
    data_pat_Svc.head()
        
    tmp = data_pat_Svc.groupby('subclass').count()
    print("Count of subclass: {}".format(len(tmp))) 
    df_gby_cls = df({'subclass': tmp.index.values})
    df_gby_cls['count_pat'] = tmp[tmp.columns[-1]].values
    df_gby_cls.head()

    df_gby_cls.describe() 
    """           count_pat
        count     727.000000
        mean     1052.097662
        std     13667.482842
        min         1.000000
        25%         2.500000
        50%        18.000000
        75%       104.000000
        max    357936.000000    
    """
    df_gby_cls.plot.hist(bins=100, log=True)
    df_gby_cls[df_gby_cls['subclass'].isin(dict_subcls_list['subclass_Svc.'])] 
    """    subclass  count_pat
        8       A01K       7010
        245     B65B       7197
        503     G01G       1328
        512     G01R      31289
        557     G06F     357936
        558     G06G       6324
        567     G06Q      46065
        571     G07B       1137
        572     G07C       2597
        574     G07F       6808
        575     G07G        557
        685     H04M      39256 """
    #occur_cutoff = df_gby_cls.describe()['count_pat']['50%']
    #occur_cutoff = df_gby_cls.describe()['count_pat']['75%']
    occur_cutoff = 500
    gby_res = df_gby_cls[df_gby_cls['count_pat']>occur_cutoff]
    gby_res.describe()
    """           count_pat
        count      85.000000
        mean     8632.752941
        std     39352.034335
        min       505.000000
        25%       777.000000
        50%      1319.000000
        75%      4070.000000
        max    357936.000000
    """
    gby_res.head()
    gby_res[gby_res['subclass'].isin(dict_subcls_list['subclass_Svc.'])]

    ####### Save subclass list
    dict_subcls_list["subclass_Tot."] = list(gby_res['subclass'].values)
    tmp = list(set(gby_res['subclass'].values)-set(dict_subcls_list['subclass_Svc.']))
    dict_subcls_list["subclass_Mfr."] = tmp
    print_count_subcls_list(dict_subcls_list)
    
    with open(WORK_DIR+'/data_json/subclass_list_res07to16.json', 'w') as f:
        json.dump(dict_subcls_list, f)
