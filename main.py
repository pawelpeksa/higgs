import logging
import sys
import threading
import pickle
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from Configuration import logger
from Configuration import Configuration as Config
from MethodsConfiguration import MethodsConfiguration
from Utils import Utils
from HiggsModels import *
from HiggsDataset import HiggsDataset

from Optimizer import determine_parameters_all

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # has to be imported before pyplot
import matplotlib.pyplot as plt

#TODO: fix indentation in all files

sess = tf.Session()


def main():
    print 'higgs 0.1'

    Config.configure_logger()

    #TOOD: extract to function
    if sys.argv[1] == 'high':
        logger().info('Using high level features')
        Config.RESULTS_DIR = './results_high/'

        Config.FEATURES_START_COL = 22 #  including
        Config.FEATURES_END_COL = 28 #  including 
    elif sys.argv[1] == 'low':
        logger().info('Using low level features')
        Config.RESULTS_DIR = './results_low/'

        Config.FEATURES_START_COL = 1 #  including
        Config.FEATURES_END_COL = 21 #  including 
    else:
        assert True, 'command line argument not specified'

    logger().info('Results directory set to ' + Config.RESULTS_DIR)
    
    Utils.maybe_create_directory(Config.RESULTS_DIR)

    higgs_fracs = Config.HIGGS_FRACS	
    
    # process_all_higgs_needed() # TODO: performance improvment possible
    	
    for higgs_frac in higgs_fracs:
        logger().info('workin on:' + str(higgs_frac) + ' data')
        higgs_data = load_data(higgs_frac)
	
        # TODO: maybe can run higgs here?
        methods_config =  determine_parameters_all(higgs_data.train.x, higgs_data.train.y, 
			                           higgs_data.valid.x, higgs_data.valid.y)

        methods_config.save(Config.RESULTS_DIR + 'methodsConfig_' + str(higgs_frac) + '.dat')	


        results = run_all_clfs(methods_config, higgs_data)
        save_results(results, higgs_frac)

        plt.figure()
        plot_from_dict(results[Config.TREE_KEY], 'tree')
        plot_from_dict(results[Config.FOREST_KEY], 'forest')
        # plot_from_dict(results[Config.SVM_KEY], 'svm')
        plot_from_dict(results[Config.ANN_KEY], 'ann')
        plot_from_dict(results[Config.DNN_KEY], 'dnn')

        file_name = Config.RESULTS_DIR + 'roc_' + str(higgs_frac) + '.pdf'
        logger().info('Saving plot at:' + file_name)
        plt.savefig(file_name)
    

    logger().info('execution finished')


def save_results(results, higgs_frac):
    with open(Config.RESULTS_DIR + 'resultDict_' + str(higgs_frac) + '.dat','w') as f:
        pickle.dump(results, f)
        

def run_all_clfs(methods_config, higgs_data):
    logger().info('Run all cfs')
    tree = DecisionTreeClassifier(max_depth=methods_config.decision_tree.max_depth)
    forest = RandomForestClassifier(max_depth=methods_config.random_forest.max_depth, 
                                    n_estimators=methods_config.random_forest.n_estimators)
    SVM = svm.SVC(kernel='linear', C=methods_config.svm.C, probability=True)
    ann = MLPClassifier(solver=methods_config.ann.solver,
                            max_iter=Config.ANN_OPIMIZER_MAX_ITERATIONS,
                            alpha=methods_config.ann.alpha,
                            hidden_layer_sizes=(MethodsConfiguration.calc_hidden_neurons(),),
                            random_state=1,
                            learning_rate='adaptive')


    threads = list()
    results = dict()

    results[Config.TREE_KEY] = []
    results[Config.FOREST_KEY] = []
    results[Config.SVM_KEY] = []
    results[Config.ANN_KEY] = []
    results[Config.DNN_KEY] = []

    threads.append(threading.Thread(target=run_clf, args=(tree, higgs_data, results[Config.TREE_KEY])))
    threads.append(threading.Thread(target=run_clf, args=(forest, higgs_data, results[Config.FOREST_KEY])))
    # threads.append(threading.Thread(target=run_clf, args=(SVM, higgs_data, results[Config.SVM_KEY])))
    threads.append(threading.Thread(target=run_clf, args=(ann, higgs_data, results[Config.ANN_KEY])))

    for thread in threads:
        thread.start()

    # run dnn on main thread
    run_higgs(higgs_data, results[Config.DNN_KEY])

    for thread in threads:
        thread.join()

    return results    


def process_all_higgs_needed():
    logger().info('Process all higgs data needed')
    higgs_fracs = Config.HIGGS_FRACS	
    for higgs_frac in higgs_fracs:
        logger().info('Loading ' + str(higgs_frac) + ' higgs')
        higgs_data = load_data(higgs_frac)


def run_clf(clf, higgs_data, result):
    logger().info('running clf:' + clf.__class__.__name__)

    clf.fit(higgs_data.train.x, higgs_data.train.y)

    prediction = clf.predict_proba(higgs_data.test.x)

    ps, ys = (prediction[:,1]).ravel(), higgs_data.test.y.ravel()
    result.append(ps)
    result.append(ys)

    logger().info('finished running clf:' + clf.__class__.__name__)


def plot_from_dict(psys, title):
    ps, ys = psys
    plot_roc(ps, ys, title)


def plot_roc(ps, ys, title):
    logger().info('Plot:' + title)
    fpr, tpr, _ = roc_curve(ys, ps)
    roc_auc = auc(fpr, tpr)

    lw = 2

    label = title + ' ROC curve (area = %0.2f)' % roc_auc

    plt.plot(fpr, tpr, lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    

reuse = None
def run_higgs(higgs_data, results):
    global reuse

    with sess.as_default():
        with tf.variable_scope('model1', reuse=reuse):
            reuse = True
            # model = HiggsLogisticRegression()
            # model = HiggsAdamBNDropoutNN(num_layers=6, size=500, keep_prob=0.9)
            model = HiggsAdamBNDropoutNN(num_layers=2, size=500, keep_prob=0.9)

        init = tf.global_variables_initializer()   
        sess.run(init)

        logistic_acus = []
	
	data_batch_train = int(math.sqrt(len(higgs_data.train.x)))
	data_batch_valid = int(math.sqrt(len(higgs_data.valid.x)))

	logger().info('Data batch train:%d Data batch valid:%d', data_batch_train, data_batch_valid)

        for i in range(25):
            logger().info('EPOCH: %d' % (i + 1))
            train(model, higgs_data.train, data_batch_train)
            ps, ys = evaluate(model, higgs_data.valid, data_batch_valid)
            valid_auc = roc_auc_score(ys, ps)
            logger().info(' VALID AUC: %.3f' % valid_auc)
            logistic_acus += [valid_auc]

        ps, ys = evaluate(model, higgs_data.valid, data_batch_valid)
        results.append(ps)
        results.append(ys)
            

def load_np_data(path):
        return np.load(path)


def higgs_data_tail(frac):
    return "_" + str(frac) + ".npy"

def load_data(data_frac):
    logger().info('loading data')

    data_dir = Config.DATA_DIR

    train_path = data_dir + "higgs_train" + higgs_data_tail(data_frac)
    valid_path = data_dir + "higgs_valid" + higgs_data_tail(data_frac)
    test_path = data_dir + "higgs_test" + higgs_data_tail(data_frac)

    train_data = None
    valid_data = None
    test_data = None

    regenerate_data = Config.REGENERATE_DATA

    logger().info('Regenerate data:' + str(regenerate_data)) 

    all_data_f = data_frac # how much data to use 
    test_data_f = Config.TEST_DATA_FRACTION # test data fraction out of entire dataset
    valid_data_f = Config.VALID_DATA_FRACTION # valid data fraction out of test dataset

    if (not (Utils.file_exist(train_path) and Utils.file_exist(valid_path) and Utils.file_exist(test_path))) or regenerate_data:

        logger().info('preparing data')
    
        df = pd.read_csv(data_dir + Config.HIGGS_ALL, header=None)
        df = df.astype(np.float32)

        data_len = len(df)

        logger().info("All data loaded: %d" % data_len)

        perm = np.random.permutation(data_len)
        all_data = df.values
        all_data = all_data[perm]

        data_len *= all_data_f
        data_len = int(data_len)
        all_data = all_data[:data_len] 

        test_data_len = int(test_data_f * data_len)

        train_data = all_data[:-(test_data_len)]

        perm = np.random.permutation(len(train_data))

        ptrain_data = train_data[perm]
        valid_data_len = int(valid_data_f * test_data_len)

        train_data = ptrain_data[:-valid_data_len]
        valid_data = ptrain_data[-valid_data_len:]
        test_data = all_data[-(test_data_len):]

        np.save(train_path, train_data)
        np.save(valid_path, valid_data)
        np.save(test_path, test_data)

    else:
        logger().info('data already prepared, loading np arrays')

        train_data = load_np_data(train_path)
        valid_data = load_np_data(valid_path)
        test_data = load_np_data(test_path)
	
    train_data, valid_data, test_data = load_columns(train_data, valid_data, test_data)

    assert train_data is not None and valid_data is not None and test_data is not None, 'data not loaded'
	
    loaded_data_len = len(train_data) + len(valid_data) + len(test_data)
    logger().info("Data len: %d with factor: %f" % (loaded_data_len, all_data_f))
    logger().info("All data should be: %d" % int(loaded_data_len/all_data_f))
    logger().info("Loading columns: %d:%d" % (Config.FEATURES_START_COL, Config.FEATURES_END_COL))

    logger().info("Train data len: %d" % len(train_data))
    logger().info("Valid data len: %d" % len(valid_data))
    logger().info("Test data len: %d" % len(test_data))    
    logger().info('data loaded')

    return HiggsDataset(train_data, valid_data, test_data)


def load_columns(train_data, valid_data, test_data):
    columns = np.arange(Config.FEATURES_START_COL, Config.FEATURES_END_COL + 1)
    columns = np.concatenate([[0],columns]) #  concatenate label

    logger().info('logging columns' + str(columns))

    train_data = train_data[:, columns]
    valid_data = valid_data[:, columns]
    test_data = test_data[:, columns]

    return train_data, valid_data, test_data


def train(model, dataset, batch_size = 16):
    epoch_size = dataset.n / batch_size
    losses = []

    for i in range(epoch_size):
        train_x, train_y = dataset.next_batch(batch_size)
        loss, _ = sess.run([model.loss, model.train_op], {model.x: train_x, model.y: train_y}) 

        losses.append(loss)
        if i % (epoch_size / 5) == 5:
            logger().info('%.2f: %.3f', i * 1.0 / epoch_size, np.mean(losses))

    return np.mean(losses)        

def evaluate(model, dataset, batch_size=32):
    dataset.shuffle()

    ps = []
    ys = []
    
    for i in range(dataset.n / batch_size):
        tx, ty = dataset.next_batch(batch_size)
        p = sess.run(model.p, {model.x: tx, model.y: ty})
        ps.append(p)
        ys.append(ty)

    ps = np.concatenate(ps).ravel()
    ys = np.concatenate(ys).ravel()       

    return ps, ys


if __name__ == '__main__':
    main()

