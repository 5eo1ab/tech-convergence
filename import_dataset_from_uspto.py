# -*- coding: utf-8 -*-
import os
print(os.getcwd())

import json
import mysql.connector
import pandas as pd
from pandas import DataFrame as df

DB_INFO = json.load(open('./data/uspto_db_connection_info.json'))
DIC_SQL = json.load(open('./data/sql-query.json'))

def set_df_result(__conn__, __query__, list_columns, out_file_nm):
	from kiie2017f import import_raw_dataset
	table_rows = import_raw_dataset.get_query_result(__conn__, __query__)
	res, f_path = df(table_rows, columns=list_columns), './data/raw_data/{}.csv'.format(out_file_nm)
	if os.path.exists(f_path):
		f_path = "{}_mod.csv".format(f_path.split('.csv')[0])
	res.to_csv(f_path, index=False)
	print(res.tail())
	return None

if __name__ == '__main__':

	cnx = mysql.connector.connect(**DB_INFO)
	print(cnx)

	obj_table = input("Which table? (t_patent[09to17]=1, t_class[target_pat]=2, t_citing[07to17]=3): ")
	if obj_table == "1":
		stmt = DIC_SQL["patent09to17"] #+ " LIMIT 10;"
		set_df_result(cnx, stmt, ['patent_no', 'issue_year'], 't_patent')
	elif obj_table == "2":
		t_patent = pd.read_csv('./data/raw_data/t_patent.csv')
		list_p_no = t_patent["patent_no"].tolist()
		print("Count of patents: ", len(list_p_no))
		stmt = DIC_SQL["pat2t_class"] + "{};".format(tuple(list_p_no))
		set_df_result(cnx, stmt, ['patent_no', '_sector', '_class', '_subclass', '_group', '_subgroup'], 't_class')
	elif obj_table == "3":
		stmt = DIC_SQL["raw_t_citing"]
		set_df_result(cnx, stmt, ['patent_no', 'cited_patent_no'], 'raw_t_citing')
	else: pass
	cnx.close()
