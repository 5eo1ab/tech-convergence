# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:26:19 2018

@author: Hanbin Seo
"""

import os
os.getcwd()
#os.chdir("D:/Paper_2018/tech-convergence")
import pandas as pd
from pandas import DataFrame as df

if __name__ == '__main__':
    
    t_class = pd.read_csv('./data/raw_data/merged_t_class.csv')
    print(t_class.head())

    period = int(input("Which period ? (09~11=1, 12~14=2, 15~17=3): "))
    t_class_by_p = t_class[t_class['period']==period][['patent_no', 'subclass']]
    print(t_class_by_p.head())
    
    matrix_by_p = pd.crosstab(t_class_by_p['patent_no'], t_class_by_p['subclass'])
    print(matrix_by_p.head())
    print(matrix_by_p.shape) # p1, p2, p3 = (612154, 1114), (829412, 920), (880695, 630)
    
    matrix_by_p.to_pickle('./data/matrix_data/matrix_W_p{}.pickle'.format(period))
    #matrix_by_p.to_csv('./data/matrix_data/matrix_W_p{}.csv'.format(period), index=False)
    
