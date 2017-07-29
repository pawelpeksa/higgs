import tensorflow as tf

def linear(x, name, size, bias=True):
    w = tf.get_variable(name + '/W', [x.get_shape()[1], size]) 
    b = tf.get_variable(name + '/b', [1, size], initializer = tf.zeros_initializer)   

    return tf.matmul(x, w) + b

class HiggsLogisticRegression(object):
    def __init__(self, lr=0.1):
        self.x = x = tf.placeholder(tf.float32, [None, 28])
        self.y = tf.placeholder(tf.float32, [None])
        x = linear(x, 'regression', 1)
        self.p = tf.nn.sigmoid(x)
        self.loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.reshape(self.y, [-1, 1]))) 
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss) 