# coding: utf-8
from sys import stdout

import numpy as np
import collections
import os
import operator
import tensorflow as tf
from senteval import SentEval
from utils import pad_sequences, dotdict

import math
import numpy
from operator import itemgetter
from numpy.linalg import norm

PATH_TO_DATA = '../../data/word-sim'

# From http://wordvectors.org/
EVALUATIONS = [
    'WS-353-ALL',  # Finkelstein et. al, 2002 (http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)
    'WS-353-SIM',  # Agirre et. al, 2009 (http://alfonseca.org/eng/research/wordsim353.html)
    'WS-353-REL',  # Agirre et. al, 2009 (http://alfonseca.org/eng/research/wordsim353.html)
    'MC-30',  # Miller and Charles, 1930 (http://www.tandfonline.com/doi/abs/10.1080/01690969108406936)
    'RG-65',  # R and G, 1965 (http://dl.acm.org/citation.cfm?id=365657)
    'RW-STANFORD',  # Luong et. al, 2013 (https://nlp.stanford.edu/~lmthang/morphoNLM/)
    'MEN-TR-3k',  # Bruni et. al, 2012 (http://clic.cimec.unitn.it/~elia.bruni/MEN.html)
    'MTurk-287',  # Radinsky et. al, 2011 (http://tx.technion.ac.il/~kirar/Datasets.html)
    'MTurk-771',  # Halawi and Dror, 2012 (http://www2.mta.ac.il/~gideon/mturk771.html)
    'YP-130',  # Yang and Powers, 2006 (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.119.1196)
    'SIMLEX-999',  # Hill et. al, 2014 (http://www.cl.cam.ac.uk/~fh295/simlex.html)
    'VERB-143',  # Baker et. al, 2014 (http://ie.technion.ac.il/~roiri/papers/63_Paper.pdf)
    'SimVerb-3500',  # Gerz et al., 2016 (http://people.ds.cam.ac.uk/dsg40/simverb.html)
]


class WordSimEvaluation:

    def __init__(self, model, dataset):
        self.m = model
        self.d = dataset

        self.summaries = []

        self.placeholders = {
            key: tf.placeholder(tf.float32, name='WordSim/%s' % key)
            for key in EVALUATIONS
        }

        self.data = {}
        self.read_data()

    def read_data(self):
        for key in EVALUATIONS:

            v = dotdict({'manual_values': {}, 'not_found': 0, 'total_size': 0})

            filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), PATH_TO_DATA, 'EN-%s.txt' % key)

            for line in open(filename, 'r'):
                wt1, wt2, value = line.strip().lower().split()
                if wt1 in self.d.vocabulary and wt2 in self.d.vocabulary:
                    w1 = self.d.vocabulary[wt1]
                    w2 = self.d.vocabulary[wt2]
                    v.manual_values[(w1, w2)] = float(value)
                else:
                    v.not_found += 1
                v.total_size += 1

            self.data[key] = v

    def evaluate(self):

        embedding_matrix = self.m.embeddings.eval()

        results = dict()

        for key, data in sorted(self.data.items(), key=operator.itemgetter(0)):

            # print('%s t: %d n: %d' % (key, data.total_size, data.not_found))

            embedding_values = dict()

            for (w1, w2), value in data.manual_values.items():
                embedding_values[(w1, w2)] = self.cosine_sim(embedding_matrix[w1], embedding_matrix[w2])

            results[key] = self.spearmans_rho(
                self.assign_ranks(data.manual_values),
                self.assign_ranks(embedding_values)
            )

        return results

    def tf_op(self):

        result = tf.stack([
            p for k, p
            in sorted(self.placeholders.items(), key=operator.itemgetter(0))
        ])

        self.summaries = [
            tf.summary.scalar('wordsim/%s' % k.lower(), p)
            for k, p in self.placeholders.items()
        ]

        return result

    def feed_dict(self, rs, feed_dict=None):
        """
        Args:
            rs: expecting tuple to be ordered by placeholder key name
        """

        if feed_dict is None:
            feed_dict = dict()

        for k, p in self.placeholders.items():
            feed_dict[p] = rs[k]

        return feed_dict

    @staticmethod
    def euclidean(vec1, vec2):

        diff = vec1 - vec2

        return math.sqrt(diff.dot(diff))

    @staticmethod
    def cosine_sim(vec1, vec2, epsilon=1e-6):

        vec1 += epsilon * numpy.ones(len(vec1))
        vec2 += epsilon * numpy.ones(len(vec1))

        return vec1.dot(vec2) / (norm(vec1) * norm(vec2))

    @staticmethod
    def assign_ranks(item_dict):
        ranked_dict = {}
        sorted_list = [(key, val) for (key, val) in sorted(item_dict.items(),
                                                           key=itemgetter(1),
                                                           reverse=True)]

        for i, (key, val) in enumerate(sorted_list):
            same_val_indices = []

            for j, (key2, val2) in enumerate(sorted_list):
                if val2 == val:
                    same_val_indices.append(j + 1)

            if len(same_val_indices) == 1:
                ranked_dict[key] = i + 1
            else:
                ranked_dict[key] = 1. * sum(same_val_indices) / len(same_val_indices)

        return ranked_dict

    @staticmethod
    def correlation(dict1, dict2):
        avg1 = 1. * sum([val for key, val in dict1.iteritems()]) / len(dict1)
        avg2 = 1. * sum([val for key, val in dict2.iteritems()]) / len(dict2)
        numr, den1, den2 = (0., 0., 0.)

        for val1, val2 in zip(dict1.itervalues(), dict2.itervalues()):
            numr += (val1 - avg1) * (val2 - avg2)
            den1 += (val1 - avg1) ** 2
            den2 += (val2 - avg2) ** 2

        return numr / math.sqrt(den1 * den2) 

    @staticmethod
    def spearmans_rho(ranked_dict1, ranked_dict2):
        assert len(ranked_dict1) == len(ranked_dict2)

        if len(ranked_dict1) == 0 or len(ranked_dict2) == 0:
            return 0.

        x_avg = 1. * sum([val for val in ranked_dict1.values()]) / len(ranked_dict1)
        y_avg = 1. * sum([val for val in ranked_dict2.values()]) / len(ranked_dict2)
        num, d_x, d_y = (0., 0., 0.)

        for key in ranked_dict1.keys():
            xi = ranked_dict1[key]
            yi = ranked_dict2[key]
            num += (xi - x_avg) * (yi - y_avg)
            d_x += (xi - x_avg) ** 2
            d_y += (yi - y_avg) ** 2

        return num / (math.sqrt(d_x * d_y))