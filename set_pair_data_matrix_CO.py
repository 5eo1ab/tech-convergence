# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 04:56:34 2018

@author: Hanbin Seo
"""

import os
os.getcwd()
#os.chdir("D:/Paper_2018/tech-convergence")
import pandas as pd
from pandas import DataFrame as df
import numpy as np

if __name__ == '__main__':

    period = int(input("Which period ? (09~11=1, 12~14=2, 15~17=3): "))
    matrix_CO = pd.read_pickle('./data/matrix_data/matrix_CO_norm_p{}.pickle'.format(period))
    print(matrix_CO.shape, type(matrix_CO))
    
    header = matrix_CO.columns.values
    matrix_CO = matrix_CO.values
    print(matrix_CO.shape, type(matrix_CO))
    
    pair_list_CO = list()
    for index, value in np.ndenumerate(matrix_CO):
        if index[0] < index[1]:
            pair_list_CO.append((index, (header[index[0]], header[index[1]]), value))
        else: continue
    print(len(pair_list_CO))

    pair_list_CO = df(pair_list_CO, columns=['index_pair', 'subclass_pair', 'CO_norm'])
    pair_list_CO.to_csv('./data/pair_data/pair_CO_norm_p{}.csv'.format(period), index=False)


