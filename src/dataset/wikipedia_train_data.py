# coding: utf-8

import numpy as np
from utils import pad_sequences
import random


PAD_SYMBOL = 0
UNK_SYMBOL = 1
SEPERATOR = ' ||| '

class WikipediaTrainData:
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
        self.ds = dataset

        self.f = open(path)

        self.vocab_size = dataset.vocab_size
        self.num_examples = len(self.keys)
        self.epochs_completed = 0

        self.x_max_length = x_max_length
        self.y_max_length = y_max_length

    def pairs(self, keys=None):
        """
        Get X for composing definitions for each separate lemma, definition combination
        also returns a negative and positive Y
        """

        x_t = [d for l, d in keys]
        x = pad_sequences(x_t, maxlen=self.x_max_length)

        y_p_t = [l for l, d in keys]
        y_p = pad_sequences(y_p_t, maxlen=self.y_max_length)

        y_n_t = [keys[random.randint(0, len(keys) - 1)][0] for _ in range(len(keys))]
        y_n = pad_sequences(y_n_t, maxlen=self.y_max_length)

        return x, y_p, y_n

    def next_batch(self, batch_size):

        keys = list()

        while len(keys) < batch_size:

            line = self.f.readline()

            if line == '': # end of file is reached
                self.epochs_completed += 1
                self.f.seek(0) # reset the file pointer
                line = self.f.readline()

            s = line.split(SEPERATOR)
            if len(s) == 2:
                l, d = s

                d = tuple([(self.ds.vocabulary[t] if t in self.ds.vocabulary else UNK_SYMBOL) for t in d.split()])
                l = tuple([(self.ds.vocabulary[t] if t in self.ds.vocabulary else UNK_SYMBOL) for t in l.split()])

                keys.append((d, l))

        return self.pairs(keys)