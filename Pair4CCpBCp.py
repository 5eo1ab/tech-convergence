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

from Matrix4Patent import Matrix4Citation
class Pair4:
    def __init__(self, period=None, matrix_type=None):
        self.period = self.__get_period__() if period is None else period
        self.matrix_type = self.__get_matrix_type__() if matrix_type is None else matrix_type
        self.idx2unit, self.unit2idx, self.dict_coor = self.__get_dict4idx__()
        self.__matrix_W__ = self.__get_df_matrix_W__()
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

        """
        res_pair = df.from_dict(self.dict_coor, orient='index')
        res_pair['coor'] = res_pair.index.values
        res_pair['x'], res_pair['y'] = res_pair['coor'].apply(lambda v: v[0]), res_pair['coor'].apply(lambda v: v[-1])
        res_pair = res_pair[res_pair[res_pair.columns[0]>0]][['x', 'y', res_pair.cloumns[0]]].sort_values(by=['x', 'y'])

        with gzip.open(path_write, 'wb') as f:
            pickle.dump({'data': res_pair.values, 'dict_header': self.idx2unit}, f)
        """
if __name__ == '__main__':
    for p_idx in range(1,4):
        p4CCp = Pair4(period=p_idx, matrix_type='CCp')
        p4CCp.run()
        p4BCp = Pair4(period=p_idx, matrix_type='BCp')
        p4BCp.run()