# coding: utf-8

import gzip
import numpy as np
from sys import stdout
from dataset import dataset


class EmbeddingProcessor:
    """
    This class can process embedding data files, load them, save them and filter them using a vocabulary
    """

    def __init__(self, vocabulary=None):

        if vocabulary is None:
            self.vocabulary = dict()
        else:
            self.vocabulary = vocabulary

        self.reversed_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

    def read_vocabulary(self, vocabulary_filename='data/compositional_wordnet/vocab.gz'):
        """
        Read an existing vocabulary, which can be saved using the Dataset class
        """

        with gzip.open(vocabulary_filename, 'r') as f:
            vocabulary_split = f.read().decode('UTF-8').split('\n')
            for i, token in enumerate(vocabulary_split):
                if token != '':
                    self.vocabulary[token] = i
                    self.reversed_vocabulary[i] = token

    def process_pretrained_embeddings(self, input_filename, output_filename, embedding_size=300):
        """
        Load the pretrained vectors, filter them using the vocabulary and save them back to file
        """

        embeddings = self.read_embeddings(input_filename, embedding_size=embedding_size, gzipped=False, print_log=True)

        return self.store_embeddings(embeddings, output_filename)

    def read_embeddings(self, input_filename, embedding_size=300, gzipped=True, print_log=False):
        """
        Read embedding file, also works if not all words in the vocab are present
        """

        embeddings = dict()
        skipped_lines = 0

        if gzipped:
            f = gzip.open(input_filename, 'r')
            lines = f.read().decode('UTF-8').split('\n')
        else:
            f = open(input_filename, 'r')
            lines = f

        for i, line in enumerate(lines):

            if print_log and i % 1000 == 0:
                stdout.write("\rReading embedding at line… %d\tfrom %s" % (i, input_filename))

            split = line.split()
            if len(split) == embedding_size + 1 and \
               split[0] in self.vocabulary and \
               self.vocabulary[split[0]] not in [dataset.PAD_SYMBOL, dataset.UNK_SYMBOL]:
                embeddings[self.vocabulary[split[0]]] = np.array([float(x) for x in split[1:]])
            else:
                skipped_lines += 1

        f.close()

        if print_log:
            print("\r%d/%d embeddings read from %s (skipped: %d)" % (
                len(embeddings), len(self.vocabulary), input_filename, skipped_lines
            ))

        return embeddings

    def store_embeddings(self, embeddings, filename='comp2vec.vec.gz'):
        """
        Load the pretrained vectors
        """

        keys = sorted(embeddings.keys())
        vocab_size = len(embeddings)
        embedding_size = len(embeddings[keys[0]])

        with gzip.open(filename, 'w') as f:
            f.write(bytes('%d %d\n' % (vocab_size, embedding_size), 'UTF-8'))
            for k in keys:
                s = ' '.join(['%f' % e for e in embeddings[k]])
                f.write(bytes('%s %s\n' % (self.reversed_vocabulary[k], s), 'UTF-8'))

        return filename
