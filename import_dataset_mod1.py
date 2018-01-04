# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:55:17 2018

@author: Hanbin Seo
"""

import os
os.getcwd()
#os.chdir("D:/Paper_2018/tech-convergence")
import pandas as pd
from pandas import DataFrame as df

def lambda_str_class(str_x):
    if len(str_x) < 2:
        return "0{}".format(str_x)
    else: return str_x
    
def lambda_split_period(issue_year):
    if issue_year < 2012:
        return 1
    elif issue_year > 2014:
        return 3
    else: return 2

if __name__ == '__main__':

    t_class = pd.read_csv('./data/raw_data/t_class.csv')
    t_class['str_class'] = t_class['_class'].map(str).apply(lambda_str_class)
    print(t_class.head())

    t_patent = pd.read_csv('./data/raw_data/t_patent.csv')
    print(t_patent.head())
   
    t_class_mod1 = t_class[['patent_no', '_sector']]
    t_class_mod1.columns = ['patent_no', 'sector']
    t_class_mod1['class'] = t_class['_sector'].map(str.upper) + t_class['str_class']
    t_class_mod1['subclass'] = t_class['_sector'].map(str.upper) + t_class['str_class'] + t_class['_subclass'].map(str.upper)
    print(t_class_mod1.tail())
    print(t_class_mod1.shape)  # (6780633, 4)
   
    t_class_mod2 = t_class_mod1.drop_duplicates(subset=['patent_no', 'subclass'], keep='first')
    print(t_class_mod2.tail())
    print(t_class_mod2.shape)  # (3692503, 4)  

    t_class_mod3 = pd.merge(t_class_mod2, t_patent, on='patent_no', how='left')
    t_class_mod3['period'] = t_class_mod3['issue_year'].apply(lambda_split_period)
    print(t_class_mod3.head())
    print(t_class_mod3.tail())
    
    t_class_mod3.to_csv('./data/raw_data/merged_t_class.csv', index=False)

