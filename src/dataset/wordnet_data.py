# coding: utf-8

import numpy as np
from utils import pad_sequences
import random


class WordnetData:
    """
    This is a Data class that can be used for a train, test or validation set,
    you can easily batch the data in a Data instance as well as return all data
    """

    def __init__(self, ds_l, ls_d, vocab_size=100000, x_max_length=32, y_max_length=16):
        """
        Args:
            ds_l: A dictionary of definitions keyed by lemma
            ls_d: A dictionary of lemmas keyed by defintions
            vocab_size: Vocabulary size
            x_max_length: Length of the maximum defintion
            y_max_length: Length of the maximum lemma
        """

        self.ds_l = ds_l
        self.ls_d = ls_d

        self.keys = list()

        self.l_idx = {l: i for i, (l, ds) in enumerate(self.ds_l.items())}
        self.idx_l = dict(zip(self.l_idx.values(), self.l_idx.keys()))

        for l, ds in ds_l.items():
            for j in range(len(ds)):
                i = self.l_idx[l]
                self.keys.append([i, j])

        self.keys = np.array(self.keys)

        self.vocab_size = vocab_size
        self.num_examples = len(self.keys)
        self.epochs_completed = 0
        self.index_in_epoch = 0

        self.x_max_length = x_max_length
        self.y_max_length = y_max_length

    def x_ls(self, n=None):
        """
        Get X for composing all unique definitions and their (multiple) associated lemmas
        used during evaluation
        """

        if n is None:
            n = len(self.ls_d)

        ds = [d for i, (d, ls) in enumerate(self.ls_d.items()) if i <= n]
        x = pad_sequences(ds, maxlen=self.x_max_length)

        lss = [ls for i, (ds, ls) in enumerate(self.ls_d.items()) if i <= n]

        return x, ds, lss

    def y_ds(self, n=None):
        """
        Get Y for composing all unique lemmas and their (multiple) associated definitions
        used during evaluation
        """

        if n is None:
            n = len(self.ds_l)

        ls = [l for i, (l, ds) in enumerate(self.ds_l.items()) if i <= n]
        y = pad_sequences(ls, maxlen=self.y_max_length)

        dss = [ds for i, (l, ds) in enumerate(self.ds_l.items()) if i <= n]

        return y, ls, dss

    def pairs(self, keys=None):
        """
        Get X for composing definitions for each separate lemma, definition combination
        also returns a negative and positive Y
        """

        if keys is None:
            keys = self.keys

        pairs = [(self.ds_l[self.idx_l[i]][j], self.idx_l[i]) for i, j in keys]

        x_t = [x for x, _ in pairs]
        x = pad_sequences(x_t, maxlen=self.x_max_length)

        y_p_t = [y for _, y in pairs]
        y_p = pad_sequences(y_p_t, maxlen=self.y_max_length)

        y_n_t = [self.idx_l[random.randint(0, len(self.idx_l) - 1)] for _ in range(len(pairs))]
        y_n = pad_sequences(y_n_t, maxlen=self.y_max_length)

        return x, y_p, y_n

    def next_batch(self, batch_size):

        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.num_examples:
            self.epochs_completed += 1

            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.keys = self.keys[perm]

            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples

        end = self.index_in_epoch

        return self.pairs(self.keys[start:end])