from Dataset import Dataset

import numpy as np

class HiggsDataset(object):
    def __init__(self, train_set, valid_set, test_set):
        self.train = Dataset(train_set[:, 1:], train_set[:, 0])
        self.valid = Dataset(valid_set[:, 1:], valid_set[:, 0])
        self.test = Dataset(test_set[:, 1:], test_set[:, 0])

    
