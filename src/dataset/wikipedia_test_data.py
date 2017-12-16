# coding: utf-8

import numpy as np
from utils import pad_sequences
import random


PAD_SYMBOL = 0
UNK_SYMBOL = 1
SEPERATOR = ' ||| '

class WikipediaTestData:
    """
    This is a Data class that can be used for a train, test or validation set,
    you can easily batch the data in a Data instance as well as return all data
    """

    def __init__(self, dataset, path, x_max_length=256, y_max_length=8):
        """
        Args:
            ds_l: A dictionary of definitions keyed by lemma
            ls_d: A dictionary of lemmas keyed by defintions
            vocab_size: Vocabulary size
            x_max_length: Length of the maximum defintion
            pad_symbol: index of the padding symbol
        """

        self.keys = list()

        with open(path) as f:

            for line in f:

                s = line.split(SEPERATOR)
                if len(s) != 2:
                    continue

                l, d = s

                d = tuple([(dataset.vocabulary[t] if t in dataset.vocabulary else UNK_SYMBOL) for t in d.split()])
                l = tuple([(dataset.vocabulary[t] if t in dataset.vocabulary else UNK_SYMBOL) for t in l.split()])

                if len(d) >= 1 and len(l) >= 1:
                    self.keys.append((l, d))

        self.vocab_size = dataset.vocab_size
        self.num_examples = len(self.keys)

        self.x_max_length = x_max_length
        self.y_max_length = y_max_length

    def x_ls(self, n=None):
        """
        Get X for composing all unique definitions and their (multiple) associated lemmas
        used during evaluation
        """

        if n is None:
            n = len(self.keys)

        ds = [d for i, (l, d) in enumerate(self.keys) if i <= n]
        x = pad_sequences(ds, maxlen=self.x_max_length)

        lss = [[l] for i, (l, d) in enumerate(self.keys) if i <= n]

        return x, ds, lss

    def y_ds(self, n=None):
        """
        Get Y for composing all unique lemmas and their (multiple) associated definitions
        used during evaluation
        """

        if n is None:
            n = len(self.keys)

        ls = [l for i, (l, d) in enumerate(self.keys) if i <= n]
        y = pad_sequences(ls, maxlen=self.y_max_length)

        dss = [[d] for i, (l, d) in enumerate(self.keys) if i <= n]

        return y, ls, dss

    def pairs(self, keys=None):
        """
        Get X for composing definitions for each separate lemma, definition combination
        also returns a negative and positive Y
        """

        if keys is None:
            keys = self.keys

        x_t = [d for l, d in keys]
        x = pad_sequences(x_t, maxlen=self.x_max_length)

        y_p_t = [l for l, d in keys]
        y_p = pad_sequences(y_p_t, maxlen=self.y_max_length)

        y_n_t = [keys[random.randint(0, len(keys) - 1)][0] for _ in range(len(keys))]
        y_n = pad_sequences(y_n_t, maxlen=self.y_max_length)

        return x, y_p, y_n