
import os
print(os.getcwd())
#os.chdir("D:/Paper_2018/tech-convergence")
import gzip, pickle
import numpy as np
import pandas as pd
from pandas import DataFrame as df

period = 3

if period > 4:
    """
    path_read = './data/model_data/np_pair_target_p{}_c5.pickle'.format(period)
    data_read = None
    with gzip.open(path_read, 'rb') as f:
        data_read = pickle.load(f)
    print(type(data_read), "\tKeys: {}".format(data_read.keys()))

    print(data_read['dict_header'])
    df_node = df(list(data_read['dict_header'].items()), columns=['id', 'label'])
    df_node['id'] = df_node['id'].apply(lambda x: 'n{}'.format(x))
    print(df_node.head())
    print("="*30)

    #print(data_read['data'], len(data_read['data']))
    #print(data_read['index_pair'], len(data_read['index_pair']))
    df_edge = df(data_read['index_pair'], columns=['source', 'target'])
    df_edge['source_lb'] = df_edge['source'].apply(lambda x: data_read['dict_header'][x])
    df_edge['target_lb'] = df_edge['target'].apply(lambda x: data_read['dict_header'][x])
    df_edge['source'] = df_edge['source'].apply(lambda x: 'n{}'.format(x))
    df_edge['target'] = df_edge['target'].apply(lambda x: 'n{}'.format(x))
    df_edge['count'] = data_read['data']
    print(df_edge.head())

    if not os.path.exists('./data/gephi_data'):
        os.mkdir('./data/gephi_data')
    path_write = './data/gephi_data/gephi_format_node_p{}.csv'.format(period+1)
    df_node.to_csv(path_write, index=False)
    path_write = './data/gephi_data/gephi_format_edge_p{}.csv'.format(period+1)
    df_edge.to_csv(path_write, index=False)
    """
else:
    data_read = None
    with gzip.open('./data/pair_data/np_pair_CO_p{}.pickle'.format(period)) as f:
        data_read = pickle.load(f)
    print(type(data_read), "\tKeys: {}".format(data_read.keys()))
    print(data_read['dict_header'])
    df_node = df(list(data_read['dict_header'].items()), columns=['id', 'label'])
    df_node['id'] = df_node['id'].apply(lambda x: 'n{}'.format(x))
    print(df_node.head())
    print("=" * 30)

    data_write = np.zeros((len(data_read['data']), 3), dtype=int)
    data_write[:, 0] = data_read['data'][:, 0].astype(int)
    data_write[:, 1] = data_read['data'][:, 1].astype(int)
    cuoff = np.percentile(data_read['data'][:, 2], 100 - 5)
    data_write[:, 2] = np.array([1 if x > cuoff else 0 for x in data_read['data'][:, 2]])
    #print(data_write)
    df_edge = df(data_write, columns=['source', 'target', 'count'])
    df_edge['source_lb'] = df_edge['source'].apply(lambda x: data_read['dict_header'][x])
    df_edge['target_lb'] = df_edge['target'].apply(lambda x: data_read['dict_header'][x])
    df_edge['source'] = df_edge['source'].apply(lambda x: 'n{}'.format(x))
    df_edge['target'] = df_edge['target'].apply(lambda x: 'n{}'.format(x))
    df_edge = df_edge[['source', 'target', 'source_lb', 'target_lb', 'count']]
    print(df_edge.head())

    if not os.path.exists('./data/gephi_data'):
        os.mkdir('./data/gephi_data')
    path_write = './data/gephi_data/gephi_format_node_p{}.csv'.format(period)
    df_node.to_csv(path_write, index=False)
    path_write = './data/gephi_data/gephi_format_edge_p{}.csv'.format(period)
    df_edge.to_csv(path_write, index=False)

