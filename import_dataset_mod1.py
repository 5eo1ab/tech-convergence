# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:55:17 2018

@author: Hanbin Seo
"""

import os
os.getcwd()

import pandas as pd
from pandas import DataFrame as df

def lambda_str_class(str_x):
    if len(str_x) < 2:
        return "0{}".format(str_x)
    else: return str_x
    
if __name__ == '__main__':

    t_class = pd.read_csv('./data/raw_data/t_class.csv')
    t_class['str_class'] = t_class['_class'].map(str).apply(lambda_str_class)
    print(t_class.head())

    t_patent = pd.read_csv('./data/raw_data/t_patent.csv')
    t_patent.head()

   
    t_class_mod1 = t_class[['patent_no', '_sector']]
    t_class_mod1.columns = ['patent_no', 'sector']
    t_class_mod1['class'] = t_class['_sector'] + t_class['str_class']
    t_class_mod1['subclass'] = t_class['_sector'] + t_class['str_class'] + t_class['_subclass']
    print(t_class_mod1.tail())
    print(t_class_mod1.shape)  # (6780633, 4)
   
    t_class_mod2 = t_class_mod1.drop_duplicates(subset=['patent_no', 'subclass'], keep='first')
    print(t_class_mod2.tail())
    print(t_class_mod2.shape)  # (3692503, 4)  

    t_class_mod3 = pd.merge(t_class_mod2, t_patent, on='patent_no', how='left')
    t_class_mod3.head()
    
    t_class_mod3.to_csv('./data/raw_data/merged_t_class.csv', index=False)

