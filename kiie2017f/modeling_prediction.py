# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 01:34:28 2017

@author: Hanbin Seo
"""

import os
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import pandas as pd
from pandas import DataFrame as df
import numpy as np

from sklearn.externals import joblib

if __name__ == '__main__':

    clf_res = joblib.load(WORK_DIR+'/clf_SVM.pkl') 
    print(clf_res)

    df_pair = None
    with open(WORK_DIR+'/dataset_model/data_subclass_norm_12to16.pickle', 'rb') as f:
        df_pair =  pickle.load(f)
    column_mn = df_pair.columns[3:].values.tolist()
    print(column_mn)

    X_data = df_pair[column_mn].values
    y_pred = clf_res.predict(X_data)
    print(X_data.shape, y_pred.shape)
        
    with open(WORK_DIR+'/dataset_model/output_result_12to16.pickle', 'rb') as f:    
        df_output = pickle.load(f)
    df_output['prediction'] = y_pred
    df_output['prediction'].value_counts()
    print(df_output['prediction'].value_counts()[1]/len(df_output['prediction']))
    df_output.head()
    
    with open(WORK_DIR+'/dataset_model/output_result_12to16.pickle', 'wb') as f: 
        pickle.dump(df_output, f)
        
        
        