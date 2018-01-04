# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 22:56:35 2018

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
    matrix_W = pd.read_pickle('./data/matrix_data/matrix_W_p{}.pickle'.format(period))
    header = matrix_W.columns.values
    print(matrix_W.head())
    print(matrix_W.shape)
    
    matrix_W = matrix_W.values
    print(type(matrix_W)) # <class 'numpy.ndarray'>
    print(np.transpose(matrix_W).shape, matrix_W.shape)
    matrix_CO = np.matmul(np.transpose(matrix_W), matrix_W)
    print(matrix_CO.shape)

    matrix_CO = df(matrix_CO, index=header, columns=header)
    matrix_CO.to_pickle('./data/matrix_data/matrix_CO_p{}.pickle'.format(period))
    