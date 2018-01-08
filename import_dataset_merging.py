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

    obj_unit = input("Which merging unit?\n(sector=1, class=2, subclass=3, group=4, subclass=5): ")
    merging_unit = {'1': 'sector', '2': 'class', '3': 'subclass', '4': 'group', '5': 'subclass'}

    t_class = pd.read_csv('./data/raw_data/t_class.csv')
    t_class['str_class'] = t_class['_class'].map(str).apply(lambda_str_class)
    t_class['str_group'] = t_class['_group'].map(str).apply(lambda_str_class)
    t_class['str_subgroup'] = t_class['_subgroup'].map(str).apply(lambda_str_class)
    print("Shape of T_CLASS:\t{}".format(t_class.shape))

    t_patent = pd.read_csv('./data/raw_data/t_patent.csv')
    print("Shape of T_PATENT:\t{}".format(t_patent.shape))

    t_class_mod1 = t_class[['patent_no', '_sector']]
    t_class_mod1.columns = ['patent_no', 'sector']
    t_class_mod1['class'] = t_class['_sector'].map(str.upper) + t_class['str_class']
    t_class_mod1['subclass'] = t_class_mod1['class'] + t_class['_subclass'].map(str.upper)
    t_class_mod1['group'] = t_class_mod1['subclass'] + ['_']*len(t_class_mod1) + t_class['str_group']
    t_class_mod1['subgroup'] = t_class_mod1['group'] + ['/']*len(t_class_mod1) + t_class['str_subgroup']
    print(t_class_mod1.shape)

    t_class_mod2 = t_class_mod1.drop_duplicates(subset=['patent_no', merging_unit[obj_unit]], keep='first')
    print(t_class_mod2.shape)
    
    t_class_mod3 = pd.merge(t_class_mod2, t_patent, on='patent_no', how='left')
    t_class_mod3['period'] = t_class_mod3['issue_year'].apply(lambda_split_period)
    print(t_class_mod3.shape)
    print(t_class_mod3.tail())

    t_class_mod3.to_csv('./data/raw_data/t_class_merging.csv'.format(), index=False)
