# coding: utf-8

import numpy as np
from utils import pad_sequences


class Data:
    """
    This is a Data class that can be used for a train, test or validation set,
    you can easily batch the data in a Data instance as well as return all data
    """

    def __init__(self, ds_l, ls_d, vocab_size=100000, max_definition_length=32):
        """
        Args:
            ds_l: A dictionary of definitions keyed by lemma
            ls_d: A dictionary of lemmas keyed by defintions
            vocab_size: Vocabulary size
            max_definition_length: Length of the maximum defintion
            pad_symbol: index of the padding symbol
        """

        self.keys = list()

        for l, ds in ds_l.items():
            for i in range(len(ds)):
                self.keys.append([l, i])

        self.keys = np.array(self.keys)

        self.ds_l = ds_l
        self.ls_d = ls_d

        self.vocab_size = vocab_size
        self.num_examples = len(self.keys)
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.max_definition_length = max_definition_length

    def x_ls(self, n=None):
        """
        Get X for composing all unique definitions and their (multiple) associated lemmas
        used during evaluation
        """

        if n is None:
            n = len(self.ls_d)

        x = pad_sequences(
            [ds for i, (ds, ls) in enumerate(self.ls_d.items()) if i <= n],
            maxlen=self.max_definition_length,
        )

        lss = [ls for i, (ds, ls) in enumerate(self.ls_d.items()) if i <= n]

        return x, lss

    def pairs(self, keys=None):
        """
        Get X for composing definitions for each separate lemma, definition combination
        also returns a negative and positive Y
        """

        if keys is None:
            keys = self.keys

        x = pad_sequences(
            [self.ds_l[k][i] for k, i in keys],
            maxlen=self.max_definition_length
        )

        y_p = [k for k, i in keys]
        y_n = np.random.randint(2, self.vocab_size, len(keys))

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