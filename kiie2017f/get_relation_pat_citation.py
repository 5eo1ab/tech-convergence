# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 20:20:32 2017

@author: Hanbin Seo
"""

import os
#import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from scipy import sparse

def get_matrix_object(df_pair_citation, object_='CC'):
    from sklearn.feature_extraction.text import CountVectorizer
    gby_key, gby_val = df_pair_citation.columns.values.tolist()[::-1]
    if object_ == 'BC':
        df_pair_citation['cited_patent_no'] = df_pair_citation['cited_patent_no'].str.replace('/', '')
        df_pair_citation = df_pair_citation[df_pair_citation['cited_patent_no'].str.isdigit()]
        df_pair_citation['cited_patent_no'] = df_pair_citation['cited_patent_no'].astype(np.int64)
        gby_key, gby_val = df_pair_citation.columns.values.tolist()
    gby_concat = df_pair_citation.groupby(gby_key)[gby_val].apply(lambda x: " ".join(x.astype(str)))
    count_vec = CountVectorizer()
    matrix_p = count_vec.fit_transform(gby_concat)
    print("Matrix Patent-(realted)Patent: ", matrix_p.shape, type(matrix_p))

    matrix_obj = sparse.triu(matrix_p.dot(matrix_p.T)) # upper trianglur matrix
    print("Matrix Patents for {}: ".format(object_), matrix_obj.shape, type(matrix_obj)) 
    
    rev_diag = np.reciprocal(matrix_obj.diagonal().astype(float)) # np.array: reverse number of diagonal value
    mat_rev_diag = sparse.diags(rev_diag, format='coo') # coo_matrix: reverse number of diagonal value
    print("Calculate: Association Strength")
    
    matrix_obj.setdiag(0)
    matrix_norm = mat_rev_diag.dot(matrix_obj.dot(mat_rev_diag))
    matrix_norm = matrix_norm.tocoo()
    print("Matrix Norm Patent for {}: ".format(object_), matrix_norm.shape, type(matrix_norm))
    return matrix_norm, matrix_obj.data


if __name__ == '__main__':
        
    period = "07to11"   #(or)     period = "12to16"
    object_type = "BC"  #(or)    object_type = "CC"
    _C_type = "citing" if object_type == "BC" else "cited"
    print("period: {}\tobject_type: {}\t".format(period, object_type))
    
    data_pat = pd.read_csv(WORK_DIR+"/res_tmp/gby_patent_{}.csv".format(period))
    df_pair_citation = pd.read_csv(WORK_DIR+"/data/citation_{}_{}.csv".format(_C_type, period))
    list_pat_no = data_pat['patent_no'].values.tolist()

    matrix_as, arr_elem_value = get_matrix_object(df_pair_citation, object_type)

    df_res = df({"_row" : matrix_as.row , 
                    "_col" :  matrix_as.col ,
                    "norm_value" : matrix_as.data })
    df_res.head()

    df_res['patent_i'] = [list_pat_no[i] for i in df_res["_row"]]
    df_res['patent_j'] = [list_pat_no[i] for i in df_res["_col"]]
    df_res['count_relation'] = [arr_elem_value[i] for i in df_res.index.values]
    df_res.head(10)

    df_res = df_res[['patent_i', 'patent_j', 'count_relation', 'norm_value']]
    df_res.head(10)

    df_res.to_csv(WORK_DIR+"/res_tmp/relation_{}_pat_{}.csv".format(object_type, period), index=False)
    print("Success!")

"""
df_import = pd.read_csv(WORK_DIR+"/data/data_relation_pat_{}_{}.csv".format(object_type, period))
df_export = df_import[df_import.columns.values.tolist()[1:]]
df_export.to_csv(WORK_DIR+"/data/data_relation_pat_{}_{}.csv".format(object_type, period), index=False)
df_export.head()
"""