# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:11:52 2018

@author: Hanbin Seo
"""

import os
print(os.getcwd())
#os.chdir("D:/Paper_2018/tech-convergence")
import pandas as pd
from pandas import DataFrame as df

if __name__ == '__main__':

    t_patent = pd.read_csv('./data/raw_data/t_patent.csv')
    print(t_patent.head())

    t_class = pd.read_csv('./data/raw_data/merged_t_class.csv')
    print(t_class.head())

    
    gby_count = t_patent.groupby('issue_year').count()
    df_gby_year = df({'issue_year': gby_count.index.values})
    df_gby_year['count_patent'] = gby_count['patent_no'].values
    print(df_gby_year.head())

    gby_count = t_class.groupby('subclass').count()['patent_no']
    df_gby_subclass = df({'subclass': gby_count.index.values})
    df_gby_subclass['count_subclass'] = gby_count.values
    print(df_gby_subclass.tail())

    df_subclass_by_period = pd.crosstab(t_class['subclass'], t_class['period'])
    df_subclass_by_period['subclass'] = df_subclass_by_period.index.values
    col_order = ['subclass'] + df_subclass_by_period.columns[:-1].tolist()
    df_subclass_by_period = df_subclass_by_period[col_order]
    print(df_subclass_by_period.head())

    df_subclass_by_year = pd.crosstab(t_class['subclass'], t_class['issue_year'])
    df_subclass_by_year['subclass'] = df_subclass_by_year.index.values
    col_order = ['subclass'] + df_subclass_by_year.columns[:-1].tolist()
    df_subclass_by_year = df_subclass_by_year[col_order]
    print(df_subclass_by_year.head())


    df_gby_year.to_csv('./data/groupby_data/gby_count_by_year.csv', index=False)
    df_gby_subclass.to_csv('./data/groupby_data/gby_count_by_subclass.csv', index=False)
    df_subclass_by_period.to_csv('./data/groupby_data/pivot_subclass_by_period.csv', index=False)
    df_subclass_by_year.to_csv('./data/groupby_data/pivot_subclass_by_year.csv', index=False)
    
