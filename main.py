import logging
import numpy as np
import sys
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score

from Configuration import Configuration as Config
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
            model = HiggsAdamBNDropoutNN(num_layers=6, size=500, keep_prob=0.9)

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


def load_data():
    logger().info('loading data')

    data_dir = Config.DATA_DIR

    # TOSO: id *.npy doesn't exist
    # test_data_f = 0.25 # test data fraction out of entire dataset
    # valid_data_f = 0.25 # valid data fraction out of test dataset


    # df = pd.read_csv(data_dir + Config.HIGGS_ALL, header=None)
    # df = df.astype(np.float32)

    # data_len = len(df)

    # test_data_len = int(test_data_f * data_len)

    # train_data = df.values[:-(test_data_len)]

    # perm = np.random.permutation(len(train_data))

    # ptrain_data = train_data[perm]
    # valid_data_len = int(valid_data_f * test_data_len)

    # train_data = ptrain_data[:-valid_data_len]
    # valid_data = ptrain_data[-valid_data_len:]
    # test_data = df.values[-(test_data_len):]

    # np.save(data_dir + "higgs_train.npy", train_data)
    # np.save(data_dir + "higgs_valid.npy", valid_data)
    # np.save(data_dir + "higgs_test.npy", test_data)

    # logger().info("Data len: %d" % data_len)
    # logger().info("Train data len: %d" % len(train_data) )
    # logger().info("Valid data len: %d" % len(valid_data))
    # logger().info("Test data len: %d" % len(test_data))

    logger().info('data loaded')

    return HiggsDataset(data_dir)

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