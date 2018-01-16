# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:11:52 2018

@author: Hanbin Seo
"""

import os
print(os.getcwd())
#os.chdir("D:/Paper_2018/tech-convergence")
import json
import pandas as pd
from pandas import DataFrame as df

DIC_UNIT = json.load(open('./data/digit-unit.json'))

if __name__ == '__main__':

    if not os.path.exists('./data/groupby_data'):
        os.mkdir('./data/groupby_data')

    is_year = input("Group by year? (yes=1, no=else): ")
    if is_year == '1':
        t_patent = pd.read_csv('./data/raw_data/t_patent.csv')
        print(t_patent.head())

        gby_count = t_patent.groupby('issue_year').count()
        df_gby_year = df({'issue_year': gby_count.index.values})
        df_gby_year['count_patent'] = gby_count['patent_no'].values
        print(df_gby_year.head())

        df_gby_year.to_csv('./data/groupby_data/gby_count_by_year.csv', index=False)
        print("Export df groupby year to CSV")
    else: pass

    obj_unit = input(DIC_UNIT['message'])
    if obj_unit not in [str(i+1) for i in range(5)]:
        import sys
        sys.exit()
    else: pass

    t_class = pd.read_csv('./data/raw_data/t_class_merging.csv')
    print(t_class.head())

    gby_count = t_class.groupby(DIC_UNIT[obj_unit]).count()['patent_no']
    df_gby_count = df({DIC_UNIT[obj_unit]: gby_count.index.values})
    df_gby_count['count'] = gby_count.values
    print(df_gby_count.tail())
    df_gby_count.to_csv('./data/groupby_data/gby_count_by_{}.csv'.format(DIC_UNIT[obj_unit]), index=False)

    df_obj_by_period = pd.crosstab(t_class[DIC_UNIT[obj_unit]], t_class['period'])
    df_obj_by_period[DIC_UNIT[obj_unit]] = df_obj_by_period.index.values
    col_order = [DIC_UNIT[obj_unit]] + df_obj_by_period.columns[:-1].tolist()
    df_obj_by_period = df_obj_by_period[col_order]
    print(df_obj_by_period.head())
    df_obj_by_period.to_csv('./data/groupby_data/pivot_{}_by_period.csv'.format(DIC_UNIT[obj_unit]), index=False)

    df_obj_by_year = pd.crosstab(t_class[DIC_UNIT[obj_unit]], t_class['issue_year'])
    df_obj_by_year[DIC_UNIT[obj_unit]] = df_obj_by_year.index.values
    col_order = [DIC_UNIT[obj_unit]] + df_obj_by_year.columns[:-1].tolist()
    df_obj_by_year = df_obj_by_year[col_order]
    print(df_obj_by_year.head())
    df_obj_by_year.to_csv('./data/groupby_data/pivot_{}_by_year.csv'.format(DIC_UNIT[obj_unit]), index=False)

    ## Summary
    df_summary = df({'{}_ALL'.format(DIC_UNIT[obj_unit]):
        df_gby_count[df_gby_count.columns[-1]].describe(),
        })
    for idx in df_obj_by_period.columns[1:]:
        key_nm = '{}_P{}'.format(DIC_UNIT[obj_unit], idx)
        df_summary[key_nm] = df_obj_by_period[df_obj_by_period[idx]>0][idx].describe()
    df_summary.to_csv('./data/groupby_data/summary_{}.csv'.format(DIC_UNIT[obj_unit]))
    