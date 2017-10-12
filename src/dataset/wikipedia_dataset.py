# coding: utf-8
import random
from collections import defaultdict
import operator
import gzip
from dataset.wikipedia_train_data import WikipediaTrainData
from dataset.wikipedia_test_data import WikipediaTestData
from utils import directory
import sys

PAD_SYMBOL = 0
UNK_SYMBOL = 1
SEPERATOR = ' ||| '

VOCAB_SIZE = 5172571

class WikipediaDataset:
    def __init__(self, x_max_length=256, y_max_length=8):

        self.x_max_length = x_max_length
        self.y_max_length = y_max_length

        # Vocabulary as defined in two dictionaries
        # self.vocabulary | token -> idx
        # self.reversed_vocabulary | idx -> token
        self.vocabulary = self.read_vocabulary()
        self.vocab_size = len(self.vocabulary)
        self.reversed_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

        self.test = WikipediaTestData(self, path=directory('/data/wikipedia/') + 'test.txt', x_max_length=x_max_length, y_max_length=y_max_length)
        self.train = WikipediaTrainData(self, path=directory('/data/wikipedia/') + 'train.txt', x_max_length=x_max_length, y_max_length=y_max_length)

    def read_vocabulary(self, path=None, vocab_min_frequency=12, vocab_size_limit=5172571):

        vocab_ns = dict()

        if path is None:
            path = directory('/data/wikipedia/') + 'lowercased.vocab'

        with open(path) as f:

            for i, line in enumerate(f):

                if i % 10000 == 0:
                    sys.stdout.write("\rReading vocabulary… %6.2f%%" % ((100 * i) / float(VOCAB_SIZE),))

                s = line.split()

                if len(s) == 2:
                    t, n = s[0], int(s[1])
                    if n >= vocab_min_frequency and i < vocab_size_limit - 2:
                        vocab_ns[t] = n
                    else:
                        break # vocabulary file is already sorted

            print("\rVocabulary read")

        # Sort the vocabulary by frequency, frequent words on top
        vocab_ns = sorted(vocab_ns.items(), key=operator.itemgetter(1), reverse=True)

        vocabulary = {token: i for i, (token, count) in enumerate(vocab_ns, 2)}
        vocabulary['PAD'] = PAD_SYMBOL
        vocabulary['UNK'] = UNK_SYMBOL

        return vocabulary

    def store_dataset(self, path='.'):
        pass