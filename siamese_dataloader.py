import numpy as np

class Dataset():
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.batch_size = batch_size

    def get_item(self, ID, emb_dim):
        X1, X2, lab, seq_len = [], [], [], []
        for id in ID:
            load = np.load(self.dir + str(id) + '.npz')
            X1 += [load['sent_1'].tolist()]
            X2 += [load['sent_2'].tolist()]
            lab += [load['lab'][0]]
            seq_len += [len(load['lab'])]

        max_len = max([len(list) for list in X1])
        for i, row in enumerate(X1):
            if len(row) <= max_len:
                X1[i] += [np.zeros(emb_dim).tolist()]*(max_len-len(row))
                X2[i] += [np.zeros(emb_dim).tolist()]*(max_len-len(X2[i]))

        return X1, X2, lab, seq_len, max_len