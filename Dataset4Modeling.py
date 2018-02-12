# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:58:47 2018

@author: Hanbin Seo
"""

import os
print(os.getcwd())
#os.chdir("D:/Paper_2018/tech-convergence")
import gzip, pickle
import numpy as np
import pandas as pd
from pandas import DataFrame as df

class Dataset4Modeling:

    def __init__(self, period, cutoff_percent):
        self.period = period
        self.cutoff = cutoff_percent # 5, 10, 20
        self.pair_bin, self.dict_header = self.__load_and_pair_bin__()
        self.dict_neighbor = dict()

    def __load_and_pair_bin__(self, load_period=0):
        load_period = self.period if load_period==0 else load_period
        with gzip.open('./data/pair_data/np_pair_CO_p{}.pickle'.format(load_period)) as f:
            data_read = pickle.load(f)
        data_write = np.zeros((len(data_read['data']), 3), dtype=int)
        data_write[:, 0] = data_read['data'][:, 0].astype(int)
        data_write[:, 1] = data_read['data'][:, 1].astype(int)
        cuoff = np.percentile(data_read['data'][:, 2], 100 - self.cutoff)
        data_write[:, 2] = np.array([1 if x > cuoff else 0 for x in data_read['data'][:, 2]])
        return data_write, data_read['dict_header']
    def __set_dict_neighbor__(self):
        df_bin = df(self.pair_bin, columns=['x_idx', 'y_idx', 'is_connect'])
        df_bin = df_bin[df_bin['is_connect']>0][['x_idx', 'y_idx']]
        for key_idx in self.dict_header.keys():
            x_neighbor = df_bin[df_bin['x_idx'] == key_idx]['y_idx'].tolist()
            y_neighbor = df_bin[df_bin['y_idx'] == key_idx]['x_idx'].tolist()
            self.dict_neighbor[key_idx] = set(x_neighbor + y_neighbor)
        return None

    def set_target_variable(self):
        path_write = './data/model_data/np_pair_target_p{}_c{}.pickle'.format(self.period, self.cutoff)
        if not os.path.exists('/'.join(path_write.split('/')[:-1])): os.mkdir('/'.join(path_write.split('/')[:-1]))
        if os.path.exists(path_write): return None
        res_data = self.__get_target_variable__()
        if res_data == None: return None
        res_dict = {'data':res_data[:,-1], 'index_pair': res_data[:,:2], 'dict_header':self.dict_header}
        with gzip.open(path_write, 'wb') as f:
            pickle.dump(res_dict, f)
        print("Length of Target Data: {}".format(len(res_data)))
        return None
    def __get_target_variable__(self):
        if self.period == 3: return None
        df_present = df(self.pair_bin, columns=[['x_idx', 'y_idx', 'is_connect']])
        df_present['x_lb'] = df_present['x_idx'].apply(lambda x: self.dict_header[x])
        df_present['y_lb'] = df_present['y_idx'].apply(lambda x: self.dict_header[x])

        pair_next, dict_next = self.__load_and_pair_bin__(load_period=self.period + 1)
        df_next = df(pair_next, columns=[['x_index', 'y_index', 'will_connect']])
        df_next['x_lb'] = df_next['x_index'].apply(lambda x: dict_next[x])
        df_next['y_lb'] = df_next['y_index'].apply(lambda x: dict_next[x])

        df_merge = df_present.merge(df_next[['x_lb', 'y_lb', 'will_connect']], how='left')
        df_merge['will_connect'] = df_merge['will_connect'].fillna(value = 0)
        df_merge['will_connect'] = df_merge['will_connect'].astype(int)
        return df_merge[['x_idx', 'y_idx', 'will_connect']].values

    def set_input_variable(self):
        path_write = './data/model_data/np_pair_input_p{}_c{}.pickle'.format(self.period, self.cutoff)
        if not os.path.exists('/'.join(path_write.split('/')[:-1])): os.mkdir('/'.join(path_write.split('/')[:-1]))
        if os.path.exists(path_write): return None
        df_citation, df_network = self.__get__citation_variable__(), self.__get_network_variable__()
        df_merge = df_citation.merge(df_network, how='inner')
        #print(df_merge.head(), "\nShape of DataFrame: {}".format(df_merge.shape))
        col_nm = df_merge.columns.tolist()[df_merge.columns.tolist().index('y_lb') + 1:]
        res_dict = {'data': df_merge[col_nm].values, 'data_norm': None,
                    'index_pair': df_merge[df_merge.columns[:2]].values, 'dict_header': self.dict_header}
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        res_dict['data_norm'] = scaler.fit_transform(res_dict['data'])
        with open(path_write, 'wb') as f:
            pickle.dump(res_dict, f)
        print("Shape of Input Data: {}".format(df_merge[col_nm].shape))
        return None
    def __get__citation_variable__(self):
        df_merge = df(self.pair_bin[:,:2], columns=[['x_idx', 'y_idx']])
        df_merge['x_lb'] = df_merge['x_idx'].apply(lambda x: self.dict_header[x])
        df_merge['y_lb'] = df_merge['y_idx'].apply(lambda x: self.dict_header[x])
        for i, obj in enumerate(['BC', 'CC', 'BCp', 'CCp']):
            with gzip.open('./data/pair_data/np_pair_{}_p{}.pickle'.format(obj, self.period), 'rb') as f:
                data_read = pickle.load(f)
            col_nm = ['x', 'y', '{}'.format(obj)] if i<2 else ['x', 'y', '{}r'.format(obj), '{}n'.format(obj)]
            df_read = df(data_read['data'], columns=[col_nm])
            df_read['x_lb'] = df_read['x'].apply(lambda x: data_read['dict_header'][x])
            df_read['y_lb'] = df_read['y'].apply(lambda x: data_read['dict_header'][x])
            df_merge = df_merge.merge(df_read[df_read.columns[2:].tolist()], how='left')
        col_nm = df_merge.columns.tolist()[df_merge.columns.tolist().index('y_lb')+1:]
        df_merge[col_nm] = df_merge[col_nm].fillna(value=0)
        #print(df_merge.head(), "\nShape of DataFrame: {}".format(df_merge.shape))
        return df_merge #{'data': df_merge[col_nm].values, 'header': df_merge.columns.tolist()}
    def __get_network_variable__(self):
        self.__set_dict_neighbor__()
        network, res_list = NetworkVariable(self.dict_neighbor), list()
        for pair in self.pair_bin:
            score = network.fit_score(pair[0], pair[1], pair[-1])
            res_list.append(score)
        col_nm = ['x_idx', 'y_idx'] + network.header
        df_res = df(np.array(res_list), columns=col_nm)
        df_res[['x_idx', 'y_idx']] = df_res[['x_idx', 'y_idx']].astype(int)
        #print(df_res.head(), "\nShape of DataFrame: {}".format(df_res.shape))
        return df_res

class NetworkVariable:
    def __init__(self, dict_neighbor):
        self.dict_neighbor = dict_neighbor
        self.__x_set__, self.__y_set__ = None, None
        self.__intersection__, self.__union__, self.__z_size_aray__ = None, None, None
        self.header = ["CN", "JC", "SI", "SC", "LHN", "HP", "HD", "PA", "AA", "RA"]

    def fit_score(self, x_idx, y_idx, is_connect):
        self.__x_set__, self.__y_set__ = self.dict_neighbor[x_idx], self.dict_neighbor[y_idx]
        self.__intersection__ = self.__x_set__ & self.__y_set__
        self.__union__ = self.__x_set__ | self.__y_set__
        self.__z_size_aray__ = [len(self.dict_neighbor[z]) for z in self.__intersection__]
        return self.__get__score__(x_idx, y_idx, is_connect)

    def __get__score__(self, x_idx, y_idx, is_connect, index_column=True):
        import math
        score_list = [x_idx, y_idx] if index_column else []
        if is_connect > 0:
            dict_score = {
                "CN": len(self.__intersection__),
                "JC": len(self.__intersection__) / len(self.__union__),
                "SI": len(self.__intersection__) / (len(self.__x_set__) + len(self.__y_set__)),
                "SC": self.__param__Dependent(0.5),
                "LHN": self.__param__Dependent(1),
                "HP": len(self.__intersection__) / min(len(self.__x_set__), len(self.__y_set__)),
                "HD": len(self.__intersection__) / max(len(self.__x_set__), len(self.__y_set__)),
                "PA": len(self.__x_set__) * len(self.__y_set__),
                "AA": sum([z_size ** (-1) for z_size in self.__z_size_aray__]),
                "RA": sum([math.log(z_size) ** (-1) for z_size in self.__z_size_aray__])
            }
            for metr in self.header: score_list.append(dict_score[metr])
        else:
            for i in range(len(self.header)): score_list.append(0)
        return score_list

    def __param__Dependent(self, lambda_value):
        mul_tmp = len(self.__x_set__) * len(self.__y_set__)
        return len(self.__intersection__) / (mul_tmp ** lambda_value)

if __name__ == '__main__':

    for p_idx in [1,2,3]:
        for c_p in [5, 10, 20]:
            print("Period={}\tCutoff percent={}%".format(p_idx, c_p))
            builder = Dataset4Modeling(p_idx, c_p)
            builder.set_target_variable()
            builder.set_input_variable()
    """p_idx, c_p = 1, 10
    builder = Dataset4Modeling(p_idx, c_p)
    builder.set_target_variable()
    builder.set_input_variable()"""
