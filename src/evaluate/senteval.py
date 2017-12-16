# coding: utf-8
from sys import stdout

import numpy as np
import operator
import logging
import os
import tensorflow as tf
from collections import defaultdict
from senteval import SentEval

from dataset.embedding_processor import EmbeddingProcessor
from utils import pad_sequences, dotdict, directory, original_embedding_file

PATH_TO_DATA = '../../data/senteval_data'

EVALUATIONS = [
    # Stanford Sentence Evaluation
    # https://nlp.stanford.edu/%7Esidaw/home/projects:nbsvm
    'MR',  # Movie Review
    'CR',  # Product Review
    'SUBJ',  # Subjectivity Status
    'MPQA',  # Opinion Polarity

    # Sentiment Analysis
    # https://nlp.stanford.edu/sentiment/index.html
    # http://www.aclweb.org/anthology/P13-1045
    'SST',

    # Question-Type Classification
    # http://cogcomp.cs.illinois.edu/Data/QA/QC/
    'TREC',

    # Paraphrase detection
    # https://www.microsoft.com/en-us/download/details.aspx?id=52398
    'MRPC',

    # Sentences Involving Compositional Knowledge
    # http://clic.cimec.unitn.it/composes/sick.html
    'SICKEntailment',
    'SICKRelatedness',

    # Semantic Textual Similarity, SemEval
    'STSBenchmark',  # http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark
    'STS12',
    'STS13',
    'STS14',  # http://alt.qcri.org/semeval2014/task10/
    'STS15',
    'STS16',
]


class SentEvalEvaluation:
    """
    Use this class to evaluate the model's parameters using SentEval's various sentence evaluation measures.
    """

    def __init__(self, model, dataset):
        self.m = model
        self.d = dataset

        self.placeholders = {
            key: tf.placeholder(tf.float32, name='SentEval/%s' % key)
            for key in EVALUATIONS
        }

        self.summaries = []

    def evaluate(self, tf_session, compose_op, pretrain=None, thorough=False):

        m = self.m
        d = self.d

        def prepare(p, samples):

            p.fallback_vocabulary = SentEvalEvaluation.create_vocabulary(samples)

            p.fallback_embeddings = SentEvalEvaluation.read_fallback_embeddings(
                p.fallback_vocabulary,
                pretrain=pretrain,
                evaluation=p.current_task
            )

            p.embedding_matrix = m.embeddings.eval()

            return

        def batcher(p, batch):

            batch = [sent if sent != [] else ['.'] for sent in batch]
            result = np.zeros((len(batch), m.embedding_size))
            idxs = list()
            embeddings = list()

            for i, wts in enumerate(batch):  # For word tokens in sentence

                embedding = np.zeros((d.x_max_length, m.embedding_size))

                j = 0
                for wt in wts:

                    wt = wt.lower()

                    if wt in self.d.vocabulary:
                        embedding[j, :] = p.embedding_matrix[self.d.vocabulary[wt], :]
                        j += 1
                    elif p.fallback_vocabulary[wt] in p.fallback_embeddings:
                        embedding[j, :] = p.fallback_embeddings[p.fallback_vocabulary[wt]]
                        j += 1
                    if j >= d.x_max_length:
                        break

                if j != 0:
                    embeddings.append(embedding)
                    idxs.append(i)

            xs = np.array(embeddings)
            cs, = tf_session.run([compose_op], m.feed_dict(xs))

            for c, idx in zip(cs, idxs):
                result[idx] = c

            return result

        results_for_tf = dict()
        results_for_out = dict()

        for evaluation in EVALUATIONS:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), PATH_TO_DATA)

            # Set params for SentEval
            params = dotdict({'task_path': path,
                              'usepytorch': self.evaluate_use_pytorch(evaluation),
                              'kfold': 10 if thorough else 2,
                              'batch_size': 64})

            # logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

            r = SentEval(params, batcher, prepare).eval(evaluation)
            result_value = 0

            try:
                if 'all' in r:
                    result_value = r['all']['pearson']['wmean']
                elif 'pearson' in r:
                    result_value = r['pearson']
                    results_for_out['%s/pearson/r' % (evaluation)] = r['pearson']
                    results_for_out['%s/spearman/r' % (evaluation)] = r['spearman']
                else:
                    result_value = r['acc']
            except KeyError:
                print('KeyError', evaluation, r)

            results_for_tf[evaluation] = result_value
            results_for_out[evaluation] = result_value

            if evaluation == 'STS12':
                self._fill_results(results_for_out, r, 'STS12', ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews'])

            if evaluation == 'STS13':
                self._fill_results(results_for_out, r, 'STS13', ['FNWN', 'OnWN', 'headlines'])

            if evaluation == 'STS14':
                self._fill_results(results_for_out, r, 'STS14', ['deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news'])

            if evaluation == 'STS15':
                self._fill_results(results_for_out, r, 'STS15', ['answers-forums', 'answers-students', 'belief', 'headlines', 'images'])

            if evaluation == 'STS16':
                self._fill_results(results_for_out, r, 'STS16', ['answer-answer', 'headlines', 'plagiarism', 'postediting', 'question-question'])

        return results_for_tf, results_for_out

    @staticmethod
    def _fill_results(results, raw, e, ses):
        for se in ses:

            results['%s/%s/pearson/r' % (e, se)] = raw[se]['pearson'][0]
            results['%s/%s/pearson/p' % (e, se)] = raw[se]['pearson'][1]
            results['%s/%s/spearman/r' % (e, se)] = raw[se]['spearman'][0]
            results['%s/%s/spearman/p' % (e, se)] = raw[se]['spearman'][1]
            results['%s/%s/n' % (e, se)] = raw[se]['nsamples']

    @staticmethod
    def create_vocabulary(sentences):
        """
        Creates vocabulary dictionary from samples
        """
        vocab_ns = defaultdict(int)

        for s in sentences:
            for t in s:
                vocab_ns[t.lower()] += 1

        vocab_ns = sorted(vocab_ns.items(), key=operator.itemgetter(1), reverse=True)

        return {token: i for i, (token, count) in enumerate(vocab_ns)}

    @staticmethod
    def read_fallback_embeddings(vocabulary, pretrain=None, evaluation='MR'):

        original_embedding_path = original_embedding_file(pretrain)

        if original_embedding_path is None:
            return dict()

        ped = EmbeddingProcessor(vocabulary)
        processed_embeddings_path = '%s/%s-%s.vec.gz' % (directory('/data/senteval_embeddings'), pretrain, evaluation)

        if not os.path.isfile(processed_embeddings_path):
            ped.process_pretrained_embeddings(input_filename=original_embedding_path,
                                              output_filename=processed_embeddings_path)

        embeddings = ped.read_embeddings(processed_embeddings_path)

        return embeddings

    @staticmethod
    def evaluate_use_pytorch(evaluation):
        """
        This static method returns TRUE if an evaluation measure should use pytorch
        """
        return evaluation in ['STSBenchmark', 'SICKRelatedness', 'SNLI']

    def tf_op(self):

        result = tf.stack([
            p for k, p
            in sorted(self.placeholders.items(), key=operator.itemgetter(0))
        ])

        self.summaries = [
            tf.summary.scalar('senteval/%s' % k.lower(), p)
            for k, p in self.placeholders.items()
        ]

        return result

    def feed_dict(self, rs, feed_dict=None):

        if feed_dict is None:
            feed_dict = dict()

        for k, p in self.placeholders.items():
            feed_dict[p] = rs[k]

        return feed_dict
