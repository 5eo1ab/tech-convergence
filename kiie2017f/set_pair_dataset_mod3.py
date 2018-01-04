# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:50:39 2017

@author: Hanbin Seo
"""

import os
import pickle
import math
os.chdir("D:\Seminar_2017_summer\Paper_fall")
WORK_DIR = os.getcwd().replace('\\','/')

import pandas as pd
from pandas import DataFrame as df
import numpy as np

class NetworkVariable:
    def __init__(self, tuple_subclass):
        self.__x_subclass__, self.__y_subclass__ = tuple_subclass
        self.__x_set__, self.__y_set__ = None, None
        self.__intersection__, self.__union__ = None, None
        self.__z_size_aray__ = None
        self.dict_score = None
        self.is_disconnect = False
    def fit(self, dict_neighbor):
        if self.__x_subclass__ not in dict_neighbor.keys() or self.__y_subclass__ not in dict_neighbor.keys():
            self.is_disconnect = True
            return None
        self.__x_set__ = dict_neighbor[self.__x_subclass__]
        self.__y_set__ = dict_neighbor[self.__y_subclass__]
        self.__intersection__ = self.__x_set__ & self.__y_set__
        self.__union__ = self.__x_set__ | self.__y_set__
        self.__z_size_aray__ = [len(dict_neighbor[z]) for z in self.__intersection__]
        return None
    def __param__Dependent(self, lambda_value):
        mul_tmp = len(self.__x_set__) * len(self.__y_set__)
        return len(self.__intersection__)/(mul_tmp**lambda_value)
    def score(self):
        if self.is_disconnect is True:
            res_list = [0]*len(["CN", "JC", "SI", "SC", "LHN", "HP", "HD", "PA", "AA", "RA"])
        else:        
            self.dict_score, res_list = {
                    "CN" : len(self.__intersection__),
                    "JC" : len(self.__intersection__)/len(self.__union__),
                    "SI" : len(self.__intersection__)/(len(self.__x_set__) + len(self.__y_set__)),
                    "SC" : self.__param__Dependent(0.5),
                    "LHN" : self.__param__Dependent(1),
                    "HP" : len(self.__intersection__)/min(len(self.__x_set__), len(self.__y_set__)),
                    "HD" : len(self.__intersection__)/max(len(self.__x_set__), len(self.__y_set__)),
                    "PA" : len(self.__x_set__) * len(self.__y_set__),
                    "AA" : sum([z_size**(-1) for z_size in self.__z_size_aray__]),
                    "RA" : sum([math.log(z_size)**(-1) for z_size in self.__z_size_aray__])
                    }, []
            for metric in ["CN", "JC", "SI", "SC", "LHN", "HP", "HD", "PA", "AA", "RA"]:
                res_list.append(self.dict_score[metric])
        return tuple(res_list)
    def printMemberVar(self):
        res_str = ">> subclass (x, y): ({}, {})\n".format(self.__x_subclass__, self.__y_subclass__)
        res_str += ">> set x: {}, {}\n".format(self.__x_set__, len(self.__x_set__))
        res_str += ">> set y: {}, {}\n".format(self.__y_set__, len(self.__y_set__))
        res_str += ">> union: {}, {}\n".format(self.__union__, len(self.__union__))
        res_str += ">> intersection: {}, {}\n".format(self.__intersection__, len(self.__intersection__))
        res_str += ">> size of intersection: {}".format(self.__z_size_aray__)
        print(res_str)
        return None
    def printScore(self):
        for k, v in self.dict_score.items():    print(">> {}\t{}".format(k, v))
        return None

if __name__ == '__main__':
    
    period = "07to11" # (or) period = "12to16"
    data_edge = pd.read_csv(WORK_DIR+'/gephi_input/input_data_edge_{}.csv'.format(period))
    data_edge = data_edge[data_edge['weight']==1]
    data_edge.head()
   
    df_pair = None
    with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'rb') as f:
        df_pair = pickle.load(f)
    df_pair.head()
    
    ####################
    gby_src = dict(data_edge.groupby('source_lb')['target_lb'].apply(lambda x: " ".join(x)))
    gby_trg = dict(data_edge.groupby('target_lb')['source_lb'].apply(lambda x: " ".join(x)))
    
    set_subcls = gby_src.keys() & gby_trg.keys()
    for key in set_subcls:  gby_src[key] += " {}".format(gby_trg[key])
    set_subcls = gby_trg.keys() - gby_src.keys()
    for key in set_subcls:  gby_src[key] = gby_trg[key]
    
    dict_neighbor = dict()
    for key, str_value in gby_src.items():
        dict_neighbor[key] = set(str_value.split())

    list_class, list_score = [], []
    metric_nm = ["score_CN", "score_JC", "score_SI", "score_SC", "score_LHN", "score_HP", 
                 "score_HD", "score_PA", "score_AA", "score_RA"]
    for pair in df_pair['pair_id'].values:
        nv = NetworkVariable(pair)
        nv.fit(dict_neighbor)
        list_score.append(nv.score())
    data_score = df(list_score, columns=metric_nm)
    data_score.head()
    
    for metric in metric_nm:
        df_pair[metric] = data_score[metric]
    df_pair.head()

    df_pair.to_csv(WORK_DIR+'/dataset_model/data_subclass_{}.csv'.format(period), index=False)
    with open(WORK_DIR+'/dataset_model/data_subclass_{}.pickle'.format(period), 'wb') as f:
        pickle.dump(df_pair, f)
    print("Success!")
