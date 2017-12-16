# coding: utf-8

import numpy as np
from utils import pad_sequences
import random
from dataset.wikipedia_train_data import WikipediaTrainData

PAD_SYMBOL = 0
UNK_SYMBOL = 1
SEPERATOR = ' ||| '

class WikipediaBucketedTrainData:
    """
    This is a Data class that can be used for a train, test or validation set,
    you can easily batch the data in a Data instance as well as return all data
    """

    def __init__(self, dataset, paths, x_max_lengths, y_max_length=8):
        """
        Args:
            ds_l: A dictionary of definitions keyed by lemma
            ls_d: A dictionary of lemmas keyed by defintions
            vocab_size: Vocabulary size
            x_max_length: Length of the maximum defintion
            pad_symbol: index of the padding symbol
        """

        self.ds = dataset

        self.buckets = list()

        for path, x_max_length in zip(paths, x_max_lengths):
            self.buckets.append(WikipediaTrainData(dataset, path=path, x_max_length=x_max_length, y_max_length=y_max_length))

        self.vocab_size = dataset.vocab_size
        self.epochs_completed = 0
        self.current_bucket = 0

        self.x_max_lengths = x_max_lengths
        self.y_max_length = y_max_length

    def next_batch(self, batch_size):

        x, y_p, y_n = self.buckets[self.current_bucket].next_batch(batch_size)
        b = self.x_max_lengths[self.current_bucket]

        self.current_bucket += 1

        if self.current_bucket % len(self.x_max_lengths) == 0:
            self.current_bucket = 0
            self.epochs_completed = self.buckets[self.current_bucket].epochs_completed

        return x, y_p, y_n, b