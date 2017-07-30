import logging
import numpy as np
import sys
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score

from Configuration import Configuration as Config
from Utils import Utils
from HiggsModels import *
from HiggsDataset import HiggsDataset

sess = tf.Session()

def main():
    print 'higgs 0.1'

    Config.configure_logger()

    higgs = load_data()

    with sess.as_default():
        with tf.variable_scope('model11', reuse=None):
            # model = HiggsLogisticRegression()
            # model = HiggsAdamBNDropoutNN(num_layers=6, size=500, keep_prob=0.9)
            model = HiggsAdamBNDropoutNN(num_layers=1, size=500, keep_prob=0.9)

        init = tf.global_variables_initializer()   

        sess.run(init)

        logistic_acus = []

        for i in range(25):
            logger().info('EPOCH: %d' % (i + 1))
            train(model, higgs.train, 8 * 1024)
            valid_auc = evaluate(model, higgs.valid, 1024)
            logger().info(' VALID AUC: %.3f' % valid_auc)
            logistic_acus += [valid_auc]

    higgs_data = logger().info('execution finished')

def load_np_data(path):
        return np.load(path)

def load_data():
    logger().info('loading data')

    data_dir = Config.DATA_DIR

    train_path = data_dir + "higgs_train.npy"
    valid_path = data_dir + "higgs_valid.npy"
    test_path = data_dir + "higgs_test.npy"

    train_data = None
    valid_data = None
    test_data = None

    regenerate_data = Config.REGENERATE_DATA
    logger().info('Regenerate data:' + str(regenerate_data)) 

    if (not (Utils.file_exist(train_path) and Utils.file_exist(valid_path) and Utils.file_exist(test_path))) or regenerate_data:

        logger().info('preparing data')

        all_data_f = Config.ALL_DATA_FRACTION  # how much data to use 
        test_data_f = Config.TEST_DATA_FRACTION # test data fraction out of entire dataset
        valid_data_f = Config.VALID_DATA_FRACTION # valid data fraction out of test dataset

        df = pd.read_csv(data_dir + Config.HIGGS_ALL, header=None)
        df = df.astype(np.float32)

        data_len = len(df)

        perm = np.random.permutation(data_len)
        all_data = df.values
        all_data = all_data[perm]

        logger().info("Taking columns: %d:%d" % (Config.FEATURES_START_COL, Config.FEATURES_END_COL))
        all_data = all_data[:, Config.FEATURES_START_COL : Config.FEATURES_END_COL + 2] # +1 first column is Y, +1 because numpy [:, a:b] takes b exclusive and we want inclusive = 1 + 1 = 2

        data_len *= all_data_f
        data_len = int(data_len)
        all_data = all_data[:data_len]

        logger().info("Data len: %d" % data_len)

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

    logger().info("Train data len: %d" % len(train_data) )
    logger().info("Valid data len: %d" % len(valid_data))
    logger().info("Test data len: %d" % len(test_data))    

    assert train_data is not None and valid_data is not None and test_data is not None, 'data not loaded'

    logger().info('data loaded')

    return HiggsDataset(train_data, valid_data, test_data)

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

    return roc_auc_score(ys, ps)


def logger():
    return logging.getLogger(Config.LOGGER_NAME)


if __name__ == '__main__':
    main()    