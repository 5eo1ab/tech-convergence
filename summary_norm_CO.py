# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:33:37 2018

@author: Hanbin Seo
"""

import pandas as pd
from pandas import DataFrame as df
from scipy import stats

from Matrix4Patent import Matrix4Patent
m4p = Matrix4Patent(auto=True)

df_summary = df()
for p_idx in range(1,4):
    pair = m4p.get_matrix(matrix_type='pair', period=p_idx)
    summ = pd.Series(pair['data'][:,2]).describe()
    summ.set_value('skew', stats.skew(pair['data'][:,2]))
    summ.set_value('kurtosis', stats.kurtosis(pair['data'][:,2]))

    summ.set_value('count_unit', len(pair['dict_header'].keys()))
    summ.set_value('count_comb', int(summ['count_unit']*(summ['count_unit']-1)/2))
    summ.set_value('ratio_exist', summ['count']/summ['count_comb'])
    df_summary['CO_norm_p{}'.format(p_idx)] = summ

df_summary.to_csv('./data/groupby_data/summary_CO_norm.csv')

