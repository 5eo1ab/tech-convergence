# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:39:32 2018

@author: Hanbin Seo
"""

import os
#os.chdir("D:/Paper_2018/tech-convergence")
import pandas as pd
import numpy as np
import gzip, pickle
from scipy import sparse as sps

class Matrix4Patent:
    def __init__(self, auto=False):
        self.__obj_unit__ = 'group' if auto == True else self.__get__obj_unit__()
    def __get__obj_unit__(self):
        import json
        DIC_UNIT = json.load(open('./data/digit-unit.json'))
        arg_value = input(DIC_UNIT['message'])
        print("object unit: {}, type: {}".format(DIC_UNIT[arg_value], type(DIC_UNIT[arg_value])))
        return DIC_UNIT[arg_value]

    def get_matrix(self, period, matrix_type='CO', is_pair=False):  # matrix_type = 'W' | 'CO' | 'Ced' | 'Cing'
        path_read = './data/matrix_data/coo_matrix_{}_p{}.pickle'.format(matrix_type, period)
        if is_pair == True:
            path_read = './data/pair_data/np_pair_{}_p{}.pickle'.format(matrix_type, period)
        #if matrix_type == 'pair': path_read = './data/matrix_data/np_pair_CO_p{}.pickle'.format(period)
        if not os.path.exists(path_read):
            if is_pair == False:
                if matrix_type == 'W': self.set_matrix_W()
                elif matrix_type == 'CO': self.set_matrix_CO()
                else: return None
            else:
                if matrix_type == 'CO': self.set_matrix_pair_CO()
                else: return None
        with gzip.open(path_read, 'rb') as f:
            data = pickle.load(f)
        if 'data' not in data.keys():
            print("shape of matrix {} = {}\nkey of dict: {}".format(matrix_type, data['data_norm'].shape, data.keys()))
        else:
            print("shape of matrix {} = {}\nkey of dict: {}".format(matrix_type, data['data'].shape, data.keys()))
        return data

    def set_matrix_W(self):
        if not os.path.exists('./data/matrix_data'):
            os.mkdir('./data/matrix_data')
        df_object = None
        for p_idx in range(1,4):
            path_write = './data/matrix_data/coo_matrix_W_p{}.pickle'.format(p_idx)
            if os.path.exists(path_write): continue
            elif df_object == None:
                df_object = pd.read_csv('./data/raw_data/t_class_merging.csv')
                print("{}\t{}".format(df_object.shape, df_object.head()))
            df_dummy = df_object[df_object['period']==p_idx][['patent_no', self.__obj_unit__]]
            with gzip.open(path_write, 'wb') as f:
                pickle.dump(self.__get_OccuranceMatrix__(df_dummy.values), f)
            print("Set matrix W, period={}".format(p_idx))
        return None
    def __get_OccuranceMatrix__(self, data):
        rows, row_pos = np.unique(data[:, 0], return_inverse=True)
        cols, col_pos = np.unique(data[:, 1], return_inverse=True)
        pivot_data = sps.coo_matrix( ([1]*len(data),(row_pos, col_pos)), shape=(len(rows), len(cols)))
        print("shape of Occurance matrix = {}".format((len(rows), len(cols))))
        return {'data': pivot_data, 'index': rows, 'column': cols}

    def set_matrix_CO(self):
        for p_idx in range(1,4):
            path_write = './data/matrix_data/coo_matrix_CO_p{}.pickle'.format(p_idx)
            if os.path.exists(path_write): continue
            matrix_W = self.get_matrix(period=p_idx, matrix_type='W')
            with gzip.open(path_write, 'wb') as f:
                pickle.dump(self.__get_CoOccuranceMatrix__(matrix_W['data'], matrix_W['column']), f)
            print("Set matrix CO, period={}".format(p_idx))
        return None
    def __get_CoOccuranceMatrix__(self, data, header):
        csr_data = data.tocsr()
        if csr_data.shape[-1] == len(header):
            res_matrix = csr_data.transpose() * csr_data
        else: res_matrix = csr_data * csr_data.transpose()
        norm_matrix = self.__get_association_strength__(res_matrix)
        print("shape of Co-Occurance matrix = {}".format(res_matrix.shape))
        return {'header': header, 'data_raw': res_matrix.tocoo(), 'data_norm': norm_matrix.tocoo()}
    def __get_association_strength__(self, csr_matrix):
        diag_matrix = sps.diags(np.reciprocal(csr_matrix.diagonal().tolist(), dtype=np.float)).tocsr()
        res_matrix = diag_matrix * csr_matrix * diag_matrix
        return res_matrix

    def set_matrix_pair_CO(self):
        if not os.path.exists('./data/pair_data'):
            os.mkdir('./data/pair_data')
        for p_idx in range(1,4):
            path_write = './data/pair_data/np_pair_CO_p{}.pickle'.format(p_idx)
            if os.path.exists(path_write): continue
            matrix = self.get_matrix(period=p_idx)
            header = dict(enumerate(matrix['header']))
            #header2index = dict((v,k) for k,v in header.items())
            pair = self.__get_matrix_pair__(matrix['data_norm'])
            with gzip.open(path_write, 'wb') as f:
                pickle.dump({'data': pair, 'dict_header':header}, f)
            print("Set pair CO, period={}".format(p_idx))
        return None
    def __get_matrix_pair__(self, coo_matrix):
        triu_data = sps.triu(coo_matrix, k=1)
        res_pair = np.array([triu_data.row, triu_data.col, triu_data.data])
        return res_pair.T

class Matrix4Citation(Matrix4Patent):
    def __init__(self):
        Matrix4Patent.__init__(self, auto=True)
        self.patent_list = None
        self.raw_t_patent = None
        self.raw_t_citing = None
    def __import_patent_list__(self, period):
        if self.raw_t_patent is None: self.__import_dataset__('t_patent')
        self.patent_list = self.raw_t_patent[self.raw_t_patent['period'] == period]['patent_no'].astype(str).tolist()
        return None
    def __import_dataset__(self, table='t_citing'):
        print("Read...{}".format(table))
        if table == 't_patent':
            from import_dataset_merging import lambda_split_period
            t_patent = pd.read_csv('./data/raw_data/t_patent.csv')
            t_patent['period'] = t_patent['issue_year'].apply(lambda_split_period)
            self.raw_t_patent = t_patent
        else: self.raw_t_citing = pd.read_csv('./data/raw_data/raw_t_citing.csv')
        return None

    def set_matrix_C(self):
        for p_idx in range(1,4):
            self.__import_patent_list__(p_idx)
            print("count of target patent: {}, period={}".format(len(self.patent_list), p_idx))
            path_write = './data/matrix_data/coo_matrix_Ced_p{}.pickle'.format(p_idx)
            if not os.path.exists(path_write):
                self.__set__matrix_C__(path_write, mode='ed')
            path_write = './data/matrix_data/coo_matrix_Cing_p{}.pickle'.format(p_idx)
            if not os.path.exists(path_write):
                self.__set__matrix_C__(path_write, mode='ing')
        return None
    def __set__matrix_C__(self, path_write, mode='ed'): # mode = 'ed' | 'ing'
        if self.raw_t_citing is None: self.__import_dataset__()
        key_export = 'cited_patent_no' if mode == 'ed' else 'patent_no'
        c_table = self.raw_t_citing[self.raw_t_citing[key_export].isin(self.patent_list)]
        c_table[key_export].astype(int)
        with gzip.open(path_write, 'wb') as f:
            pickle.dump(self.__get_OccuranceMatrix__(c_table.values), f)
        print("Set matrix C{}".format(mode))
        return None

    def set_matrix_CCp_BCp(self):
        for p_idx in range(1,4):
            matrix_C = self.get_matrix(period=p_idx, matrix_type='Ced')
            path_write = './data/matrix_data/coo_matrix_CCp_p{}.pickle'.format(p_idx) # CC by patent
            if not os.path.exists(path_write):
                with gzip.open(path_write, 'wb') as f:
                    pickle.dump(self.__get_CoOccuranceMatrix__(matrix_C['data'], matrix_C['column']), f)
                print("Set matrix CCp, period={}".format(p_idx))
            matrix_C = self.get_matrix(period=p_idx, matrix_type='Cing')
            path_write = './data/matrix_data/coo_matrix_BCp_p{}.pickle'.format(p_idx) # BC by patent
            if not os.path.exists(path_write):
                with gzip.open(path_write, 'wb') as f:
                    pickle.dump(self.__get_CoOccuranceMatrix__(matrix_C['data'], matrix_C['index']), f)
                print("Set matrix BCp, period={}".format(p_idx))
        return None

    def set_matrix(self):
        return None

    def __get_matrix_A__(self, period):
        matrix_Ced = self.get_matrix(period, matrix_type='Ced')
        matrix_W = self.get_matrix(period, matrix_type='W')

if __name__ == '__main__':

    matrix = Matrix4Patent(auto=True)
    matrix.set_matrix_W()
    matrix.set_matrix_CO()
    matrix.set_matrix_pair_CO()

    #citation = Matrix4Citation()
    #citation.set_matrix_C()
    #citation.set_matrix_CCp_BCp()