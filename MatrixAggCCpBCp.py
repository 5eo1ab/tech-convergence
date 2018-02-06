# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:45:34 2018

@author: Hanbin Seo
"""

import os
#os.chdir("D:/Paper_2018/tech-convergence")
import numpy as np
from pandas import DataFrame as df
import gzip, pickle
from scipy import sparse as sps

from Matrix4Patent import Matrix4Citation
class Pair4CCpBCp:
    def __init__(self, period=None, matrix_type=None):
        self.period = self.__get_period__() if period is None else period
        self.matrix_type = self.__get_matrix_type__() if matrix_type is None else matrix_type
        self.idx2unit, self.unit2idx, self.dict_coor = self.__get_dict4idx__()
        self.__matrix_W__ = None
        self.pair, self.idx2pat = self.__get_pair_and_Index2PatentNo__()

    def __get_period__(self):
        p_value = input("Which period ? (1, 2, 3): ")
        return int(p_value)
    def __get_matrix_type__(self):
        mt = input("Which matrix ? (CCp=1, BCp=2): ")
        dict_tmp = {'1': 'CCp', '2': 'BCp'}
        return dict_tmp[mt]

    def __get_dict4idx__(self):
        mc = Matrix4Citation()
        dict_idx2unit = mc.get_matrix(period=self.period, matrix_type='CO', is_pair=True)['dict_header']
        dict_unit2idx = dict((v, k) for k, v in dict_idx2unit.items())
        dict_coor = dict((k, 0) for k in self.__generate_coor__(len(dict_idx2unit)))
        return dict_idx2unit, dict_unit2idx, dict_coor
    def __generate_coor__(self, size):
        x = 0
        while x < size:
            y = x + 1
            while y < size:
                yield (x, y)
                y += 1
            x += 1
        return None

    def __get_df_matrix_W__(self):
        mc = Matrix4Citation()
        matrix_W = mc.get_matrix(period=self.period, matrix_type='W')
        pair_W = mc.__get_matrix_pair__(matrix_W['data'])
        dict_idx_W, dict_col_W = dict(enumerate(matrix_W['index'])), dict(enumerate(matrix_W['column']))
        df_W = df(np.array([
            [dict_idx_W[arr[0]], dict_col_W[arr[1]]]
            for arr in pair_W]), columns=['patent_no', 'unit'])
        df_W['patent_no'] = df_W['patent_no'].astype(int)
        print(df_W.head())
        return df_W

    def __get_pair_and_Index2PatentNo__(self):
        mc = Matrix4Citation()
        matrix_CC = mc.get_matrix(period=self.period, matrix_type=self.matrix_type)
        pair_CC = mc.__get_matrix_pair__(matrix_CC['data_raw'])
        dict_key_CC = dict(enumerate(matrix_CC['header'].astype(int)))
        return pair_CC, dict_key_CC

    def run(self):
        #path_write = "./data/pair_data/np_pair_{}_p{}.pickle".format(self.matrix_type, self.period)
        path_write = "./data/pair_data/TMP_dict_{}_p{}.pickle".format(self.matrix_type, self.period)
        if os.path.exists(path_write): return None
        self.__matrix_W__ = self.__get_df_matrix_W__()
        from itertools import product
        size = len(self.pair)
        for i, row in enumerate(self.pair):
            list_x = self.__matrix_W__[self.__matrix_W__['patent_no'] == self.idx2pat[row[0]]]['unit'].tolist()
            list_y = self.__matrix_W__[self.__matrix_W__['patent_no'] == self.idx2pat[row[1]]]['unit'].tolist()
            for x, y in product(list_x, list_y):
                if x == y: continue
                pair = (self.unit2idx[x], self.unit2idx[y])
                if pair[0] < pair[-1]: self.dict_coor[pair] += row[-1]
                else: self.dict_coor[(pair[-1], pair[0])] += row[-1]
            print("{}/{}\t(p, mt)=({}, {})".format(i+1, size, self.period, self.matrix_type))
        with gzip.open(path_write, 'wb') as f:
            pickle.dump(self.dict_coor, f)
        return None

    def reshape(self):
        path_write = './data/pair_data/np_pair_{}_p{}.pickle'.format(self.matrix_type, self.period)
        if os.path.exists(path_write): return None
        dict_pair, path_read = None, './data/pair_data/TMP_dict_{}_p{}.pickle'.format(self.matrix_type, self.period)
        if not os.path.exists(path_read): self.run()
        with gzip.open(path_read, 'rb') as f:
            dict_pair = pickle.load(f)
        print("Read TEMP dict_{}, period={},\nsize of dict: {}".format(self.matrix_type, self.period, len(dict_pair)))

        df_pair = df.from_dict(dict_pair, orient='index')
        df_pair['coor'] = df_pair.index.values
        df_pair['x'] = df_pair['coor'].apply(lambda c: c[0])
        df_pair['y'] = df_pair['coor'].apply(lambda c: c[-1])
        val_col_nm = df_pair.columns[0]
        df_pair = df_pair[df_pair[val_col_nm] > 0][['x', 'y', val_col_nm]].sort_values(by=['x', 'y'])
        print("result shape of pair: {}".format(df_pair.shape))

        with gzip.open(path_write, 'wb') as f:
            #pickle.dump({'data': mc.__get_matrix_pair__(coo_mat), 'dict_header': self.idx2unit}, f)
            pickle.dump({'data': df_pair.values, 'dict_header': self.idx2unit}, f)
        print("Set pair {}, period={}".format(self.matrix_type, self.period))
        print("="*25)
        return None

    def normalization(self):
        path_io = './data/pair_data/np_pair_{}_p{}.pickle'.format(self.matrix_type, self.period)
        if not os.path.exists(path_io): self.reshape()
        data_read, data_write, new_key_nm = None, dict(), 'dict_count'
        with gzip.open(path_io, 'rb') as f:
            data_read = pickle.load(f)
        if new_key_nm in data_read.keys(): return None
        print("Key of dict: {}".format(data_read.keys()))
        print("Shape of data: {}".format(data_read['data'].shape))

        df_data = df(data_read['data'], columns=['x', 'y', 'c_ij'])
        data_write[new_key_nm], res_arr = dict(), list()
        for idx in data_read['dict_header'].keys():
            cnt = df_data[df_data['x']==idx].append(df_data[df_data['y']==idx])['c_ij'].sum()
            if cnt > 0: data_write[new_key_nm][idx] = cnt
        for i, j, c_ij in df_data.values:
            ww = data_write[new_key_nm][i] * data_write[new_key_nm][j]
            res_arr.append(c_ij/ww)
        df_data['norm'] = res_arr

        data_write['data'] = df_data.values
        data_write['dict_header'], data_write['data_raw'] = data_read['dict_header'], data_read['data']
        print("Key of dict: {}".format(data_write.keys()))
        print("Shape of data: {}".format(data_write['data'].shape))

        with gzip.open(path_io, 'wb') as f:
            pickle.dump(data_write, f)
        print("Set pair {} with Association Strength, period={}".format(self.matrix_type, self.period))
        print("=" * 25)
        return None

if __name__ == '__main__':
    for p_idx in range(1,4):
        p4CCp = Pair4CCpBCp(period=p_idx, matrix_type='CCp')
        #p4CCp.run()
        #p4CCp.reshape()
        p4CCp.normalization()
        p4BCp = Pair4CCpBCp(period=p_idx, matrix_type='BCp')
        #p4BCp.run()
        #p4BCp.reshape()
        p4BCp.normalization()
