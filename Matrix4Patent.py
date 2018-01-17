# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:39:32 2018

@author: Hanbin Seo
"""

import os
import pandas as pd
import numpy as np
import gzip, pickle
from scipy import sparse as sps

class Matrix4Patent:
    def __init__(self):
        self.__obj_unit__ = self.__get__obj_unit__()

    def __get__obj_unit__(self):
        import json
        DIC_UNIT = json.load(open('./data/digit-unit.json'))
        arg_value = input(DIC_UNIT['message'])
        print("object unit: {}, type: {}".format(DIC_UNIT[arg_value], type(DIC_UNIT[arg_value])))
        return DIC_UNIT[arg_value]

    def get_matrix(self, matrix_type='W', period=1):
        # matrix_type = 'W' | 'CO'
        path_read = './data/matrix_data/coo_matrix_{}_p{}.pickle'.format(matrix_type, period)
        if not os.path.exists(path_read):
            self.set_matrix_W()
        with gzip.open(path_read, 'rb') as f:
            data = pickle.load(f)
        return data

    def set_matrix_W(self):
        if not os.path.exists('./data/matrix_data'):
            os.mkdir('./data/matrix_data')
        df_object = pd.read_csv('./data/raw_data/t_class_merging.csv')
        print("{}\t{}".format(df_object.shape,df_object.head()))
        for p_idx in range(1,4):
            path_write = './data/matrix_data/coo_matrix_W_p{}.pickle'.format(p_idx)
            if os.path.exists(path_write): continue
            df_dummy = df_object[df_object['period']==p_idx][['patent_no', self.__obj_unit__]]
            with gzip.open(path_write, 'wb') as f:
                pickle.dump(self.__get_OccuranceMatrix__(df_dummy.values), f)
            print("Set matrix W, period={}".format(p_idx))
        return None
    def __get_OccuranceMatrix__(self, data):
        rows, row_pos = np.unique(data[:, 0], return_inverse=True)
        cols, col_pos = np.unique(data[:, 1], return_inverse=True)
        pivot_data = sps.coo_matrix( ([1]*len(data),(row_pos, col_pos)), shape=(len(rows), len(cols)))
        print("shape of matrix W = {}".format((len(rows), len(cols))))
        return {'data': pivot_data, 'index': rows, 'column': cols}

    def set_matrix_CO(self):
        for p_idx in range(1,4):
            path_write = './data/matrix_data/coo_matrix_CO_p{}.pickle'.format(p_idx)
            if os.path.exists(path_write): continue

    def __get_CoOccuranceMatrix__(self):


if __name__ == '__main__':
    matrix = Matrix4Patent()
    matrix.set_matrix_W()