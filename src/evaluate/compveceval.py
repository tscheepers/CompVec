# coding: utf-8
from sys import stdout

from sklearn.neighbors.ball_tree import BallTree
import numpy as np
import tensorflow as tf


class CompVecEvalEvaluation:

    def __init__(self, model, data):
        self.m = model
        self.d = data

        # CompositionalWordNet Ranking
        # Analyzing the compositional properties of word embeddings (Scheepers et al. 2017)
        self.mnr = tf.placeholder(tf.float32, name='CompVecEval/MNR')  # Mean Normalized Rank
        self.mrr = tf.placeholder(tf.float32, name='CompVecEval/MRR')  # Mean Reciprocal Rank
        self.map = tf.placeholder(tf.float32, name='CompVecEval/MAP')  # Mean Average Precision
        self.mpa10 = tf.placeholder(tf.float32, name='CompVecEval/MP_10')  # Mean Precision@10

        self.summaries = []

    def evaluate(self, tf_session, compose_x_op, compose_y_op, n_definitions=None, batch_size=512):

        # Compose all definitions
        x, ds, lss = self.d.x_ls(n_definitions)
        xc = None

        for x_batch in np.array_split(x, batch_size, axis=0):
            xc_batch, = tf_session.run([compose_x_op], self.m.feed_dict(x=x_batch))
            if xc is None:
                xc = np.array(xc_batch)
            else:
                xc = np.concatenate((xc, xc_batch), axis=0)

        y, ls, dss = self.d.y_ds(n_definitions)
        yc = None

        for y_batch in np.array_split(y, batch_size, axis=0):
            yc_batch, = tf_session.run([compose_y_op], self.m.feed_dict(y_p=y_batch))
            if yc is None:
                yc = np.array(yc_batch)
            else:
                yc = np.concatenate((yc, yc_batch), axis=0)

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
            ranking_array = np.array([(1.0 if i in lsm else 0.0) for i in ranking])
            rankings.append(ranking_array)

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
            tf.summary.scalar('compveceval/mnr', self.mnr),
            tf.summary.scalar('compveceval/mrr', self.mrr),
            tf.summary.scalar('compveceval/map', self.map),
            tf.summary.scalar('compveceval/mp_10', self.mpa10),
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