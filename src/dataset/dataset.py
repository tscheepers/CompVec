# coding: utf-8
import random
from collections import defaultdict
import operator
import gzip
from dataset.data import Data
from dataset.wordnet_fetcher import WordnetFetcher


PAD_SYMBOL = 0
UNK_SYMBOL = 1
SEPERATOR = '|||'


class Dataset:
    def __init__(self, vocab_limit=1000000, test_split=0.05, max_definition_length=32):

        self.max_definition_length = max_definition_length

        wordnet = WordnetFetcher()

        # fetch definitions
        ds, d_vocab_ns = wordnet.fetch_definitions()

        # fetch target words per definitions
        ls, l_vocab_ns = wordnet.fetch_lemmas(ds)

        # Sort the vocabulary by frequency, frequent words on top
        vocab_ns = sorted(d_vocab_ns.items(), key=operator.itemgetter(1), reverse=True)

        # Vocabulary as defined in two dictionaries
        # self.vocabulary | token -> idx
        # self.reversed_vocabulary | idx -> token
        self.vocabulary = {token: i for i, (token, count) in enumerate(vocab_ns, 2) if i <= vocab_limit}
        self.vocabulary['PAD'] = PAD_SYMBOL
        self.vocabulary['UNK'] = UNK_SYMBOL
        self.vocab_size = len(self.vocabulary)
        self.reversed_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

        # These are the words with actual definitions
        self.targets = sorted([self.vocabulary[token] for token, _ in l_vocab_ns.items() if token in self.vocabulary])
        self.target_vocabulary = {self.reversed_vocabulary[t]: t for t in self.targets}
        self.reversed_target_vocabulary = {t: self.reversed_vocabulary[t] for t in self.targets}

        data = self._data(ds, ls)

        self.test, self.train = self._data_split([test_split, (1.0 - test_split)], data)

    def _data(self, ds, ls):
        """
        Build the definition dictionary, with a list of definitions per target word
        """

        data = list()

        for (l_id, l_ts) in ls.items():
            l_t = l_ts[0]  # We only fetch one token lemmas
            if l_t in self.vocabulary:
                l = self.vocabulary[l_t]
                d = tuple([(self.vocabulary[d_t] if d_t in self.vocabulary else UNK_SYMBOL) for d_t in ds[l_id]])
                data.append((l, d))

        return data

    def _data_split(self, ps, data):
        """
        We create a good split for test/train/val sets
        """

        # The sum of the chances should be 1
        assert sum(ps) == 1
        ps = sorted(ps)

        # Split groups
        gs = [
            (p, list(), defaultdict(list), defaultdict(list), defaultdict(int))
            for p in ps
        ]

        # We shuffle the data
        random.seed(0)
        random.shuffle(data)

        # Take a data point from the shuffled set one at a time
        for i, x in enumerate(data):

            # Data point
            (l, d) = x

            # Now we check the groups to add the datapoint to
            # Shuffle gs for randomization
            for g, (p, xs, ds_l, ls_d, d_ns) in enumerate(sorted(gs, key=lambda *args: random.random())):

                # Function to add the data point to the group/split
                def add_x():
                    xs.append(x)
                    ds_l[l] += [d]
                    ls_d[d] += [l]
                    for t in d: d_ns[t] += 1

                # We add double definitions and lemmas to the same group
                if l in ds_l or d in ls_d:
                    add_x()
                    break

                # We make sure train group contain definition tokens for lemmas of other groups
                found = False
                if p == ps[-1]:  # train group has the larges split prob
                    for t in d:
                        if t in self.reversed_target_vocabulary and t not in ds_l and d_ns[t] < 1:
                            found = True
                            break

                # Lastly we try to adhere to the split defined in ps
                if found or i * p >= len(xs) or g == len(gs) - 1:
                    add_x()
                    break

        return tuple([
             Data(ds_l, ls_d, self.vocab_size, self.max_definition_length)
             for _, _, ds_l, ls_d, _
             in gs
         ])

    def store_dataset(self, path='.'):
        """
        Stores the entire dataset
        """
        self.store_data(self.train, filename=path + '/train_data.gz')
        self.store_data(self.test, filename=path + '/test_data.gz')
        self.store_vocab(filename=path + '/vocab.gz')
        self.store_vocab(filename=path + '/vocab.tsv', tsv=True)

    def store_data(self, data=None, filename='train_data.gz'):
        """
        Store the definition and lemma data pairs
        """

        if data is None:
            data = self.train

        with gzip.open(filename, 'w') as f:
            for k, i in data.keys:
                d = ' '.join(['%s' % self.reversed_vocabulary[t] for t in data.ds_l[k][i]])
                f.write(bytes('%s %s %s\n' % (self.reversed_vocabulary[k], SEPERATOR, d), 'UTF-8'))

        return filename

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
