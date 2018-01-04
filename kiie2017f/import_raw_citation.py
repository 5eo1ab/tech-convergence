# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:22:27 2017

@author: Hanbin Seo
"""

import os
import json
import pickle
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import mysql.connector
from pandas import DataFrame as df
from import_raw_dataset import get_query_result, set_query_result

if __name__ == '__main__':

    DB_CONN_INFO = json.load(open(WORK_DIR+"/data_json/uspto_db_connection_info.json"))
    cnx = mysql.connector.connect(**DB_CONN_INFO)
    print(cnx)
    
    query_ = "SELECT patent_no, cited_patent_no FROM T_CITING "
    table_rows = get_query_result(cnx, query_)
    data_citation = df(table_rows, columns=['patent_no', 'cited_patent_no'])
    data_citation.head()
    cnx.close()

    path = '/'.join(WORK_DIR.split('/')[:-1])
    with open(path+"/df_raw_citation_v201706.pickle", "wb") as f:
        pickle.dump(data_citation, f)
    print("Success!")