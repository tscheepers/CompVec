# coding: utf-8
from sys import stdout

from sklearn.neighbors.ball_tree import BallTree
import numpy as np
import tensorflow as tf


class NNEvaluation:

    def __init__(self, model, dataset):
        self.m = model
        self.d = dataset

        # CompositionalWordNet Ranking
        # Analyzing the compositional properties of word embeddings (Scheepers et al. 2017)
        self.mnr = tf.placeholder(tf.float32, name='NN/MNR')  # Mean Normalized Rank
        self.mrr = tf.placeholder(tf.float32, name='NN/MRR')  # Mean Reciprocal Rank
        self.map = tf.placeholder(tf.float32, name='NN/MAP')  # Mean Average Precision
        self.mpa10 = tf.placeholder(tf.float32, name='NN/MP_10')  # Mean Precision@10

        self.summaries = []

    def evaluate(self, tf_session, compose_x_op, compose_y_op, n_definitions=None):

        # Compose all definitions
        x, ds, lss = self.d.test.x_ls(n_definitions)
        xc, = tf_session.run([compose_x_op], self.m.feed_dict(x=x))
        xc = np.array(xc)

        y, ls, dss = self.d.test.y_ds(n_definitions)
        yc, = tf_session.run([compose_y_op], self.m.feed_dict(y_p=y))
        yc = np.array(yc)

        # Rank all definitions
        r = self.rank(xc, yc, ls, lss)

        # Calculate metrics
        metrics = self.calculate_metrics(r)

        return metrics

    def rank(self, cs, yc, ls, lss):

        targets = {l: i for (i, l) in enumerate(ls)}

        # Number of results (lemmas) ranked
        n_results = len(yc)

        # Build ball tree model
        ball_tree = BallTree(yc)

        rs = ball_tree.query(cs, k=n_results, return_distance=False)

        rankings = list()

        for i, (ranking, ls) in enumerate(zip(rs, lss)):

            lsm = [targets[l] for l in ls]
            rankings.append(np.array([(1.0 if i in lsm else 0.0) for i in ranking]))

        return rankings

    def calculate_metrics(self, rankings):

        # Total number of rankings
        n = len(rankings)

        # Calculate ranking metrics
        rrs = np.zeros(n)  # reciprocal rank
        nrs = np.zeros(n)  # normalized rank
        aps = np.zeros(n)  # average precision
        pa10s = np.zeros(n)  # precision@10

        for i, ranking in enumerate(rankings):

            relevant_idxs, = np.where(ranking == 1)
            first_relevant_idx = float(relevant_idxs[0]) + 1.0

            nrs[i] = first_relevant_idx / float(len(ranking))
            rrs[i] = 1.0 / first_relevant_idx
            aps[i] = self.average_precision(ranking)
            pa10s[i] = self.precision_at_k(ranking, 10)

        return {
            'MNR': np.mean(nrs),
            'MRR': np.mean(rrs) * 100,
            'MAP': np.mean(aps) * 100,
            'MP@10': np.mean(pa10s) * 100
        }

    def tf_op(self):

        result = tf.stack((self.mnr,  self.mrr, self.map, self.mpa10))

        self.summaries = [
            tf.summary.scalar('nn/mnr', self.mnr),
            tf.summary.scalar('nn/mrr', self.mrr),
            tf.summary.scalar('nn/map', self.map),
            tf.summary.scalar('nn/mp_10', self.mpa10),
        ]

        return result

    def feed_dict(self, r, feed_dict=None):

        if feed_dict is None:
            feed_dict = dict()

        feed_dict[self.mnr] = r['MNR']
        feed_dict[self.mrr] = r['MRR']
        feed_dict[self.map] = r['MAP']
        feed_dict[self.mpa10] = r['MP@10']

        return feed_dict

    @staticmethod
    def precision_at_k(r, k):
        """
        Args:
            r: A relevance ranking e.g. [ 0 1 1 0 0 0 ]
            k: At k e.g. 3
        """

        return np.sum(r[:k]) / np.float(k)

    @staticmethod
    def average_precision(r):
        """
        Args:
            r: A relevance ranking e.g. [ 0 1 1 0 0 0 ]
        """
        precision_at_k = np.add.accumulate(r) / np.arange(1, r.shape[0] + 1, dtype=np.float)
        return np.sum(precision_at_k * r) / np.sum(r, dtype=np.float)