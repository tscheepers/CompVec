# coding: utf-8
import random
from collections import defaultdict
import operator
import gzip
from dataset.wikipedia_bucketed_train_data import WikipediaBucketedTrainData
from dataset.wikipedia_test_data import WikipediaTestData
from utils import directory
import sys

PAD_SYMBOL = 0
UNK_SYMBOL = 1
SEPERATOR = ' ||| '

VOCAB_SIZE = 5172571

class WikipediaDataset:
    """
    An instance of the WikipediaDataset can create a nice train, test split for the larger Wikipedia dataset.
    The train data is split into 4 separate buckets with different x_max_lengths.
    """

    def __init__(self, x_max_lengths=None, y_max_length=8):

        self.name = 'wikipedia'

        self.x_max_length = max(x_max_lengths)
        self.x_max_lengths = x_max_lengths
        self.y_max_length = y_max_length

        # Vocabulary as defined in two dictionaries
        # self.vocabulary | token -> idx
        # self.reversed_vocabulary | idx -> token
        self.vocabulary = self.read_vocabulary()
        self.vocab_size = len(self.vocabulary)
        self.reversed_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

        self.test = WikipediaTestData(self, path=directory('/data/wikipedia_data/') + 'test.txt',
                                      x_max_length=self.x_max_length, y_max_length=self.y_max_length)

        self.train = WikipediaBucketedTrainData(self, paths=
        [
            directory('/data/wikipedia_data/') + 'train-33.txt',
            directory('/data/wikipedia_data/') + 'train-62.txt',
            directory('/data/wikipedia_data/') + 'train-118.txt',
            directory('/data/wikipedia_data/') + 'train-504.txt'
        ], x_max_lengths=self.x_max_lengths, y_max_length=self.y_max_length)

    def read_vocabulary(self, path=None, vocab_min_frequency=18, vocab_size_limit=5171164):
        """
        Read vocabulary from vocab file.
        """

        vocab_ns = dict()

        if path is None:
            path = directory('/data/wikipedia_data/') + 'vocab.txt'

        with open(path) as f:

            for i, line in enumerate(f):

                if i % 10000 == 0:
                    sys.stdout.write("\rReading vocabulary… %6.2f%%" % ((100 * i) / float(VOCAB_SIZE),))

                s = line.split()

                if len(s) != 2:
                    continue

                t, n = s[0], int(s[1])
                if n >= vocab_min_frequency and i < vocab_size_limit - 2:
                    vocab_ns[t] = n
                else:
                    break # vocabulary file is already sorted

            print("\rVocabulary read %d" % (len(vocab_ns)+2))

        # Sort the vocabulary by frequency, frequent words on top
        vocab_ns = sorted(vocab_ns.items(), key=operator.itemgetter(1), reverse=True)

        vocabulary = {token: i for i, (token, count) in enumerate(vocab_ns, 2)}
        vocabulary['PAD'] = PAD_SYMBOL
        vocabulary['UNK'] = UNK_SYMBOL

        return vocabulary

    def store_dataset(self, path='.'):
        """
        Stores the entire dataset
        """
        self.store_vocab(filename=path + '/vocab.gz')
        self.store_vocab(filename=path + '/vocab.tsv', tsv=True)

    def store_vocab(self, filename='vocab.gz', tsv=False):
        """
        Store vocabulary
        tsv: Tab seperated value for TensorBoard embedding visualization
        """

        if tsv:
            with open(filename, 'w') as f:
                # f.write('Word\n')
                for i in range(self.vocab_size):
                    t = self.reversed_vocabulary[i]
                    f.write('%s\n' % (t,))

        else:
            with gzip.open(filename, 'w') as f:
                for i in range(self.vocab_size):
                    t = self.reversed_vocabulary[i]
                    f.write(bytes('%s\n' % (t,), 'UTF-8'))

        return filename