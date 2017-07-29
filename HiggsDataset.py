from Dataset import Dataset

import numpy as np

def load_data(path):
        return np.load(path)

class HiggsDataset(object):
    def __init__(self, data_dir):
        data = load_data(data_dir + "/higgs_train.npy")
        self.train = Dataset(data[:, 1:], data[:, 0])

        data = load_data(data_dir + "/higgs_valid.npy")
        self.valid = Dataset(data[:, 1:], data[:, 0])

        data = load_data(data_dir + "/higgs_test.npy")
        self.test = Dataset(data[:, 1:], data[:, 0])

    
