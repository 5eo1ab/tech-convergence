# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 01:25:04 2018

@author: Hanbin Seo
"""

import os
os.getcwd()
#os.chdir("D:/Paper_2018/tech-convergence")
import pandas as pd
from pandas import DataFrame as df

def get_matrix_association_strenth(np_matrix):
    import numpy as np
    mat_w_rev = np.diag(np.reciprocal(np.diag(np_matrix), dtype=float))
    mat_c_ij = np.triu(np_matrix)
    res_matrix = np.matmul(np.matmul(mat_w_rev, mat_c_ij), mat_w_rev)
    return res_matrix

if __name__ == '__main__':
    
    period = int(input("Which period ? (09~11=1, 12~14=2, 15~17=3): "))
    matrix_CO = pd.read_pickle('./data/matrix_data/matrix_CO_p{}.pickle'.format(period))
    print(matrix_CO.shape, type(matrix_CO))
    
    header = matrix_CO.columns.values
    matrix_CO = matrix_CO.values
    print(matrix_CO.shape, type(matrix_CO))

    matrix_CO_norm = get_matrix_association_strenth(matrix_CO)
    print(matrix_CO)
    print(matrix_CO_norm)

    matrix_CO_norm = df(matrix_CO_norm, index=header, columns=header)
    matrix_CO_norm.head()
    
    matrix_CO_norm.to_pickle('./data/matrix_data/matrix_CO_norm_p{}.pickle'.format(period))
    