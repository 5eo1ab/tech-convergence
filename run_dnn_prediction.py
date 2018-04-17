import os
#print(os.getcwd())
import numpy as np
__SEED__ = 10
np.random.seed(__SEED__)
import tensorflow as tf
import gzip, pickle

def load_data_pred():
    with gzip.open('./data/model_data/np_pair_input_p{}_c{}.pickle'.format(2, 5), 'rb') as f:
        data_read = pickle.load(f)
    print("Keys of read data: {}".format(data_read.keys()))
    print("Shape of Input data: {}".format(data_read['data_norm'].shape))
    #return data_read['data_norm']
    return data_read['data_norm'], {'index_pair': data_read['index_pair'], 'dict_header': data_read['dict_header']}

def predict_dnn():
    X_data, y_dict = load_data_pred()
    n_input, n_classes = X_data.shape[-1], 2
    n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5 = 300, 300, 300, 300, 300
    ## Setting placeholder
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder("float")
    ## Setting variable
    initializer = tf.contrib.layers.xavier_initializer(seed=__SEED__)
    weights = {
        'h1': tf.Variable(initializer([n_input, n_hidden1])),
        'h2': tf.Variable(initializer([n_hidden1, n_hidden2])),
        'h3': tf.Variable(initializer([n_hidden2, n_hidden3])),
        'h4': tf.Variable(initializer([n_hidden3, n_hidden4])),
        'h5': tf.Variable(initializer([n_hidden4, n_hidden5])),
        'out': tf.Variable(initializer([n_hidden5, n_classes])),
    }
    bias = {
        'b1': tf.Variable(initializer([n_hidden1])),
        'b2': tf.Variable(initializer([n_hidden2])),
        'b3': tf.Variable(initializer([n_hidden3])),
        'b4': tf.Variable(initializer([n_hidden4])),
        'b5': tf.Variable(initializer([n_hidden5])),
        'out': tf.Variable(initializer([n_classes]))
    }
    from run_dnn_modeling import __model_multilayer_perceptron__
    pred = __model_multilayer_perceptron__(x, weights, bias, 1.0)
    res_pred = tf.argmax(pred, axis=1)

    load_saver = tf.train.Saver()
    with tf.Session() as sess:
        load_saver.restore(sess, './tmp/model_c5.ckpt')
        print("Model restored!")
        res = sess.run(res_pred, feed_dict={x:X_data})
        print("Result: {},\tCount of result: {},\tType of result: {}".format(res, len(res), type(res)))
    y_dict['data'] = res
    return y_dict

if __name__ == '__main__':

    y_data = predict_dnn()
    print("Keys of y_data: {}".format(y_data.keys()))

    path_write = './data/result_pred_data'
    if not os.path.exists(path_write):
        os.mkdir(path_write)
    with gzip.open(path_write+'/np_pair_prediction_c5.pickle', 'wb') as f:
        pickle.dump(y_data, f)

