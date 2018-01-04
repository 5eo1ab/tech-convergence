# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 22:43:03 2017

@author: Hanbin Seo
"""

import os
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import pandas as pd
from pandas import DataFrame as df
import numpy as np

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE

def get_score_dict(y_true, y_pred):
    res_dic = {'accuracy': metrics.accuracy_score(y_true, y_pred),
                'precision': metrics.precision_score(y_true, y_pred),
                'recall': metrics.recall_score(y_true, y_pred),
                'f1_score': metrics.f1_score(y_true, y_pred)
                }
    return res_dic

if __name__ == '__main__':

    df_pair = None
    with open(WORK_DIR+'/dataset_model/data_subclass_norm_07to11.pickle', 'rb') as f:
        df_pair =  pickle.load(f)
    column_mn = df_pair.columns[3:].values.tolist()
    print(column_mn)

    ### remove citation variables
    print(column_mn)
    print(column_mn[4:])
    column_mn = column_mn[4:]    
    
    df_output = None
    with open(WORK_DIR+'/dataset_model/output_result_07to11.pickle', 'rb') as f:
        df_output = pickle.load(f)
    y_data = df_output['true_12to16'].values
    df_output.head()
    
    #col_input = column_mn
    X_data = df_pair[column_mn].values
    print(X_data.shape, y_data.shape)


    ################
    sm = SMOTE(kind='svm', k_neighbors=3)
    X_smote, y_smote = sm.fit_sample(X_data, y_data)
    print(X_smote.shape, y_smote.shape)
    """
    X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.10, random_state=42, stratify=y_data)
    X_smote, y_smote = sm.fit_sample(X_train, y_train)
    print(X_smote.shape, y_smote.shape)
    """
    
    param_svm = {
            'C' : [10**i for i in range(-3, 4)],
            'gamma': [10**i for i in range(-3, 4)], # except 'linear' kernel
            #'class_weight' : [{1:w} for w in [0.5, 1, 1.5]]
            }
    skf = StratifiedKFold(n_splits=5, random_state=10)
    clf  = svm.SVC(kernel='rbf', random_state=10)
    #fbeta_scorer = make_scorer(fbeta_score, beta=0.5)
    grid_search = GridSearchCV(clf, param_svm, scoring = make_scorer(fbeta_score, beta=0.1), 
                               cv = skf, n_jobs = -1, verbose = 1)
    print("fitting...")
    grid_search.fit(X_smote, y_smote)
    clf_best = grid_search.best_estimator_
    print("best score: {} \nAt...{}".format(grid_search.best_score_, clf_best))
    df_res = df(grid_search.cv_results_)
    df_res.sort_values(by='rank_test_score', inplace=True)
    y_pred = grid_search.predict(X_data)
    dic_scores = get_score_dict(y_data, y_pred)
    dic_scores

    joblib.dump(clf_best, WORK_DIR+'/clf_SVM.pkl')
    df_res.to_csv(WORK_DIR+'/clf_result_v2.csv', index=False)
    
    