import os
#print(os.getcwd())
import numpy as np
__SEED__ = 10
np.random.seed(__SEED__)
import tensorflow as tf

period = 1
#period, cutoff_percent = 1, 10

def load_data_modeling(one_of_hot=True):
    print("Cutoff: {}".format(cutoff_percent))
    import gzip, pickle
    ## loading dataset
    X_train, y_train, X_test, y_test = None, None, None, None
    X_sm, y_sm, y_train_oh, y_test_oh = None, None, None, None
    for p in range(2):
        with gzip.open('./data/model_data/np_pair_input_p{}_c{}.pickle'.format(p+1, cutoff_percent), 'rb') as f:
            data_read1 = pickle.load(f)
        with gzip.open('./data/model_data/np_pair_target_p{}_c{}.pickle'.format(p+1, cutoff_percent), 'rb') as f:
            data_read2 = pickle.load(f)
        if p<1: X_train, y_train = data_read1['data_norm'], data_read2['data']
        else: X_test, y_test = data_read1['data_norm'], data_read2['data']
    print("="*30)
    print("Key of read data:\n>> Input: {}\n>> Output: {}".format(data_read1.keys(), data_read2.keys()))
    print("Column of Input data: {}".format(data_read1['columns']))
    print("Shape of load data @ init.:\n>> Train Raw\tInput: {}\tOutput: {}".format(X_train.shape, y_train.shape))
    print(">> Test Raw\tInput: {}\tOutput: {}".format(X_test.shape, y_test.shape))
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(k_neighbors=3, n_jobs=-1, random_state=__SEED__)
    X_sm, y_sm = sm.fit_sample(X_train, y_train)
    if not one_of_hot: return X_sm, y_sm, X_test, y_test
    y_train_oh, y_test_oh = np.zeros((len(y_sm), 2)), np.zeros((len(y_test), 2))
    idx_pos, idx_neg = np.where(y_sm > 0), np.where(y_sm < 1)
    y_train_oh[idx_neg, 0], y_train_oh[idx_pos, 1] = 1., 1.
    idx_pos, idx_neg = np.where(y_test > 0), np.where(y_test < 1)
    y_test_oh[idx_neg, 0], y_test_oh[idx_pos, 1] = 1., 1.
    print("Shape of load data @ fin.:\n>> Train with SMOTE, OneOfHotCoding\tInput: {}\tOutput: {}".format(X_sm.shape, y_train_oh.shape))
    print(">> Test with OneOfHotCoding\tInput: {}\tOutput: {}".format(X_test.shape, y_test_oh.shape))
    print("=" * 30)
    return X_sm, y_train_oh, X_test, y_test_oh

def learning_dnn(feed_dict=None, print_step=True):
    if feed_dict == None:
        learning_rate, training_iteration, iteration_per_epoch, batch_size = 0.001, 10**5, 10**3, 100
    else:
        learning_rate, training_iteration, iteration_per_epoch, batch_size = \
            feed_dict['learning_rate'], feed_dict['training_iteration'], \
            feed_dict['iteration_per_epoch'], feed_dict['batch_size']
    X_train, y_train, X_test, y_test = load_data_modeling()
    n_input, n_classes = X_train.shape[-1], y_train.shape[-1]
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
    ## Construct model
    pred = __model_multilayer_perceptron__(x, weights, bias, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ## Setting model accuracy
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    ## Running model
    init, saver = tf.global_variables_initializer(), tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        for iter in range(training_iteration):
            idx_list = np.random.choice(len(y_train), batch_size).tolist()
            batch_x, batch_y = X_train[idx_list], y_train[idx_list]
            if iter % iteration_per_epoch == 0:
                train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                idx_list = np.random.choice(len(y_test), batch_size).tolist()
                test_acc = sess.run(accuracy, feed_dict={x: X_test[idx_list], y: y_test[idx_list], keep_prob: 1.0})
                print("Step: {}/{}\tlr: {}\t training accuracy: {:.8f} \t test accuracy: {:.8f}".format(
                    iter // iteration_per_epoch + 1, training_iteration // iteration_per_epoch,
                    learning_rate, train_acc, test_acc))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
        print("=" * 30)
        if feed_dict != None: print("Hyper Parameter: {}".format(feed_dict))
        else:
            print("Hyper Parameter: default value")
            print("learning_rate, training_iteration, iteration_per_epoch, batch_size = ({},{},{},{})".format(
                0.001, 10 ** 5, 10 ** 3, 100 ))
        y_p, y_true = tf.argmax(pred, 1), np.argmax(y_test, 1)
        _, y_pred = sess.run([accuracy, y_p], feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        score = __get_dict_score__(y_true, y_pred)

        print("Test Score: {}".format(score))

        _ = saver.save(sess, './tmp/model_c{}.ckpt'.format(cutoff_percent))
    return None

def __model_multilayer_perceptron__(x, weights, biases, keep_prob):
    tf.set_random_seed = __SEED__
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1'])), keep_prob)
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), keep_prob)
    layer_3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])), keep_prob)
    layer_4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])), keep_prob)
    layer_5 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])), keep_prob)
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
    return out_layer
def __get_dict_score__(y_true, y_pred):
    from sklearn import metrics
    score = {
        'Accuracy': metrics.accuracy_score(y_true, y_pred),
        'Precision': metrics.precision_score(y_true, y_pred),
        'Recall': metrics.recall_score(y_true, y_pred),
        'F1_score': metrics.f1_score(y_true, y_pred),
        'ROC_AUC': metrics.roc_auc_score(y_true, y_pred)
    }
    return score

def sub_lr():
    X_train, y_train, X_test, y_test = load_data(one_of_hot=False)
    from sklearn import linear_model
    logistic = linear_model.LogisticRegression()
    logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)
    print(__get_dict_score__(y_test, y_pred))
    return None

def sub_rf():
    X_train, y_train, X_test, y_test = load_data(one_of_hot=False)
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print(__get_dict_score__(y_test, y_pred))
    return None

def sub_svm():
    X_train, y_train, X_test, y_test = load_data(one_of_hot=False)
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    idx_neg_train, idx_neg_test = np.where(y_train<1), np.where(y_test<1)
    y_train[idx_neg_train], y_test[idx_neg_test] = -1, -1
    from sklearn import svm
    clf = svm.SVC(kernel='rbf', random_state=__SEED__)
    clf.fit(X_train, y_train)
    print("Fitted!")
    y_pred = clf.predict(X_test)
    print(__get_dict_score__(y_test, y_pred))
    return None

if __name__ == '__main__':

    from random import choice
    cutoff_percent = 5
    hyper_param = {
        'learning_rate': 0.001 ,
        'training_iteration': 10 ** 5,
        'iteration_per_epoch': 10 ** 3,
        'batch_size': 100
    }
    learning_dnn(hyper_param)

    """
    for c in [5, 10, 15, 20, 25]:
        cutoff_percent = c
        #sub_lr()
        #sub_svm()
        #sub_rf()
    from random import choice
    for c in [5, 10, 15, 20, 25]:
        cutoff_percent = c
        hyper_param = {
            'learning_rate': 0.001 ** choice(range(-4, -3+1)),
            'training_iteration': 10 ** choice(range(5, 6+1)),
            'iteration_per_epoch': 10 ** 3,
            'batch_size': 100
        }
        learning_dnn(hyper_param)
    """


