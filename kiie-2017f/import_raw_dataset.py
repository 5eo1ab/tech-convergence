# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 20:51:32 2017

@author: Hanbin Seo
"""

import os
import json
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import mysql.connector
import pandas as pd
from pandas import DataFrame as df

def get_query_result(cnx, parm_query):
    cursor = cnx.cursor()
    cursor.execute(parm_query)
    table_rows = cursor.fetchall()
    cursor.close()
    return table_rows

def set_query_result(cnx, parm_query):
    cursor = cnx.cursor()
    cursor.execute(parm_query)
    cursor.close()
    cnx.commit()
    return None

if __name__ == '__main__':
    
    DB_CONN_INFO = json.load(open(WORK_DIR+"/data_json/uspto_db_connection_info.json"))
    cnx = mysql.connector.connect(**DB_CONN_INFO)
    print(cnx)

    query_ = "SELECT DISTINCT patent_no, sector, class, subclass "
    query_ += "FROM T_CLASS " 
    query_ += "WHERE class_type='IPC' AND patent_no IN "
    query_ += "(SELECT patent_no FROM T_Patent "
    query_ += "WHERE issue_year>2006 AND issue_year<2012)"
    #query_ += "WHERE issue_year>2011 AND issue_year<2017)"
    print(query_)

    table_rows = get_query_result(query_)
    data = df(table_rows, columns=['patent_no', '_sector', '_class', '_subclass'])
    cnx.close()
    
    data['class'] = ["".join(ele) for ele in 
        data[['_sector', '_class']].values.tolist()]
    data['subclass'] = ["".join(ele) for ele in 
        data[['_sector', '_class', '_subclass']].values.tolist()]
    data.head()
    
    data.to_csv(WORK_DIR+'/data/data_2007to2011.csv', index=False)
    #data.to_csv(WORK_DIR+'/data/data_2012to2016.csv', index=False)
