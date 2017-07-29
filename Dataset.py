import numpy as np

class Dataset(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.n = x.shape[0]
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        self.x = self.x[perm]
        self.y = self.y[perm]           

        self._next_id = 0    

    def next_batch(self, batch_size):
        if self._next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self._next_id        
        self._next_id += batch_size
        end_id = cur_id+batch_size

        return self.x[cur_id : end_id], self.y[cur_id : end_id]        