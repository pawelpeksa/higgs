import tensorflow as tf
from Configuration import Configuration as Config


def parameters_n():
    return Config.FEATURES_END_COL - Config.FEATURES_START_COL + 1


def linear(x, name, size, bias=True):
    w = tf.get_variable(name + '/W', [x.get_shape()[1], size]) 
    b = tf.get_variable(name + '/b', [1, size], initializer = tf.zeros_initializer)   

    return tf.matmul(x, w) + b

def batch_norm(x, name):
    mean, var = tf.nn.moments(x, [0])
    normalized_x = (x-mean) * tf.rsqrt(var + 1e-8)
    gamma = tf.get_variable(name + '/gamma', [x.get_shape()[-1]], initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable(name + '/beta', [x.get_shape()[-1]]) 
    return gamma * normalized_x + beta  


class HiggsLogisticRegression(object):
    def __init__(self, lr=0.1):
        self.x = x = tf.placeholder(tf.float32, [None, parameters_n])
        self.y = tf.placeholder(tf.float32, [None])
        x = linear(x, 'regression', 1)
        self.p = tf.nn.sigmoid(x)
        self.loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.reshape(self.y, [-1, 1]))) 
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss) 


class HiggsNeuralNetwork(object):
    def __init__(self, num_layers=1, size=100, lr=0.1):
        self.x = x = tf.placeholder(tf.float32, [None, parameters_n])
        self.y = tf.placeholder(tf.float32, [None])

        for i in range(num_layers):
            x = tf.nn.relu(linear(x, 'linear_%d' % i, size))

        x = linear(x, 'regression', 1)
        self.p = tf.nn.sigmoid(x)
        self.loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.reshape(self.y, [-1, 1])))
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)


class HiggsBNNeuralNetwork(object):
    def __init__(self, num_layers=1, size=100, lr=0.1):
        self.x = x = tf.placeholder(tf.float32, [None, parameters_n])
        self.y = tf.placeholder(tf.float32, [None])

        for i in range(num_layers):
            x = tf.nn.relu(batch_norm(linear(x, 'linear_%d' % i, size), 'bn_%d' % i))

        x = linear(x, 'regression', 1)
        self.p = tf.nn.sigmoid(x)
        self.loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.reshape(self.y, [-1, 1])))
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)


class HiggsAdamBNNeuralNetwork(object):
    def __init__(self, num_layers=1, size=100, lr=0.1):
        self.x = x = tf.placeholder(tf.float32, [None, parameters_n])
        self.y = tf.placeholder(tf.float32, [None])

        for i in range(num_layers):
            x = tf.nn.relu(batch_norm(linear(x, 'linear_%d' % i, size), 'bn_%d' % i))

        x = linear(x, 'regression', 1)
        self.p = tf.nn.sigmoid(x)
        self.loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.reshape(self.y, [-1, 1])))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)        


class HiggsAdamBNDropoutNN(object):

    def __init__(self, num_layers=1, size=100, lr=0.1, keep_prob=1.0):
        self.x = x = tf.placeholder(tf.float32, [None, parameters_n()])
        self.y = tf.placeholder(tf.float32, [None])

        for i in range(num_layers):
            x = tf.nn.relu(batch_norm(linear(x, 'linear_%d' % i, size), 'bn_%d' % i))
            if keep_prob < 1.0:
                x = tf.nn.dropout(x, keep_prob)

        x = linear(x, 'regression', 1)
        self.p = tf.nn.sigmoid(x)
        self.loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.reshape(self.y, [-1, 1])))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)                


