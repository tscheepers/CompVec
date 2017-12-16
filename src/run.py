# coding: utf-8

import tensorflow as tf
import math
import time
import pickle
from collections import defaultdict
from evaluate.evaluate import Evaluation
from model import Model
from tensorflow.contrib.tensorboard.plugins import projector
from utils import directory


class Run:
    """
    Train a specific model
    """

    def __init__(self,
                 run_group_name,
                 dataset,
                 embedding_processor,
                 yc=True,
                 pretraining=None,
                 learning_rate=1e-3,
                 batch_size=512,
                 embedding_size=300,
                 stop_gradients_y_n=False,
                 dropout_keep_p=0.75,
                 margin=0.25,
                 composition='sum',
                 loss='mse',
                 refine_after_x_steps=0,
                 no=None):

        self.dataset = dataset
        self.embedding_processor = embedding_processor
        self.y_composition = yc
        self.no = no

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_keep_p = dropout_keep_p
        self.pretraining = pretraining
        self.refine_after_x_steps = refine_after_x_steps

        self.run_dir = directory('/out/run-%s' % run_group_name, ['logs', 'tsne', 'embeddings', 'output'])
        self.data_dir = embedding_processor.path

        self.x_max_buckets = None

        if hasattr(self.dataset, 'x_max_lengths'):
            self.x_max_buckets = self.dataset.x_max_lengths

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Model
            self.model = Model(
                embedding_size=embedding_size,
                x_max_buckets=self.x_max_buckets,
                x_max_length=self.dataset.x_max_length,
                y_max_length=self.dataset.y_max_length,
                y_composition=self.y_composition,
                vocab_size=self.dataset.vocab_size,
                margin=margin,
                composition=composition,
                loss=loss,
                stop_gradients_y_n=stop_gradients_y_n
            )

            # Evaluator
            self.evaluation = Evaluation(self.model, self.dataset)

            # Assign Tensorflow Operations
            self.assign_ops()

        # Setup Writers for Tensorboard
        self.setup_writers()

    @property
    def name(self):
        """
        Get the run name
        """
        return '%s-%s%s-%s-d%.2f-m%.2f-sg%d-lr%.3f-bs%d-mx%d-my%d-em%d-r%d-yc%d-%s' % (
            self.dataset.name, self.model.composition, ('' if self.no is None else '-%d' % self.no),
            ('random' if self.pretraining is None else self.pretraining),
            self.dropout_keep_p, self.model.margin, int(self.model.stop_gradients_y_n),
            self.learning_rate, self.batch_size, self.dataset.x_max_length, self.dataset.y_max_length,
            self.model.embedding_size, self.refine_after_x_steps, (1 if self.y_composition else 0), self.model.loss
        )

    # noinspection PyAttributeOutsideInit
    def assign_ops(self):
        """
        Assign all the Tensorflow operations required to run the training and testing of the model
        """

        # Optimizer
        if self.x_max_buckets is None:
            self.loss_op, self.compose_op, self.compose_e_op, self.compose_y_op = self.model.fn()
            self.optimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)
        else:
            self.loss_op, self.compose_op, self.compose_e_op, self.compose_y_op = self.model.fn()
            self.optimize_ops = dict()
            self.loss_ops = dict()
            self.summary_loss_ops = dict()

            for bucket in self.x_max_buckets:
                loss_op, _, _, _ = self.model.fn(reuse=True, bucket=bucket)
                self.loss_ops[bucket] = loss_op
                self.summary_loss_ops[bucket] = tf.summary.merge(self.model.bucketed_summaries[bucket])
                with tf.variable_scope("optimizer", reuse=(bucket != self.x_max_buckets[0])):
                    self.optimize_ops[bucket] = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op)

        # Add variable initializer
        self.init_op = tf.global_variables_initializer()

        # Summary
        self.compveceval_op = self.evaluation.compveceval.tf_op()
        self.senteval_op = self.evaluation.senteval.tf_op()
        self.wordsim_op = self.evaluation.wordsim.tf_op()

        self.summary_loss_op = tf.summary.merge(self.model.summaries)
        self.summary_loss_evaluation_op = tf.summary.merge(
            self.model.summaries +
            self.evaluation.compveceval.summaries +
            self.evaluation.wordsim.summaries +
            self.evaluation.senteval.summaries
        )

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    # noinspection PyAttributeOutsideInit
    def setup_writers(self, add_embedding_visualization=True):
        """
        Setup the writers for Tensorboard Logging, also setup embedding visualization in Tensorboard
        """

        # For writing data to TensorBoard
        self.train_writer = tf.summary.FileWriter('%s/logs/%s/train' % (self.run_dir, self.name),
                                                  self.graph)
        self.test_writer = tf.summary.FileWriter('%s/logs/%s/test' % (self.run_dir, self.name),
                                                 self.graph)

        # For visualizing the embeddings in TensorBoard
        if add_embedding_visualization:
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = self.model.embeddings.name
            embedding.metadata_path = '%s/vocab.tsv' % (self.data_dir,)
            projector.visualize_embeddings(self.test_writer, config)

    def initialize_model(self):
        """
        Initialize the model's embeddings randomly, or with pretrained embeddings
        """

        # We must initialize all variables before we use them.
        self.session.run(self.init_op)

        # If pretraining is applied we assign embedding values for the tokens we have embeddings for
        if self.pretraining is not None:

            embeddings = self.embedding_processor.read_embeddings('%s/%s.vec.gz' % (self.data_dir, self.pretraining))
            embeddings_matrix = self.model.embeddings.eval(session=self.session)

            for i, embedding in embeddings.items():
                embeddings_matrix[i, :] = embedding

            assign_op = self.model.embeddings.assign(embeddings_matrix)
            self.session.run(assign_op)

            del embeddings
            print('Pre-trained embeddings from %s loaded' % self.pretraining)

    # noinspection PyAttributeOutsideInit
    def execute(self, steps=20000, rank_every_x_steps=2500):
        """
        Run the model
        """

        self.start_time = time.time()

        self.steps = steps
        self.average_loss = 0.0
        self.out_attr = dict()
        self.out_lines = list()
        self.store_attr = defaultdict(dict)

        # Only allocate a fraction of the GPU, since there needs to be enough room for SentEval PyTorch
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        # Start Tensorflow Session
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        with self.session.as_default():

            # Initialize the model
            self.initialize_model()

            for step in range(steps + 1):
                self.step = step
                if not self.execute_step(rank_every_x_steps=rank_every_x_steps): break

    def execute_step(self, rank_every_x_steps=2500):
        """
        Run one of the model
        """

        test_result = True  # If test is not executed, all is alright
        if self.every_x_steps(x=500):
            test_result = self.test_step(rank_every_x_steps=rank_every_x_steps)

        # Note that this is expensive (~20% slowdown if computed every 1000 steps)
        # if self.every_x_steps(x=2500):
        #     self.out(self.evaluation.log_similarity())

        if self.every_x_steps(x=self.steps, include_first_step=False):
            path = self.evaluation.plot_with_labels(filename='%s/tsne/%s-%d.pdf' % (self.run_dir, self.name, self.step))
            self.out("tSNE plot of step %d saved in file: %s" % (self.step, path))

        if self.every_x_steps(x=self.steps, include_first_step=False):
            self.save_model()

        train_result = self.train_step(out_every_x_steps=100)

        if len(self.out_attr) + len(self.out_lines) > 0:
            self.print_out()

        return test_result and train_result

    def every_x_steps(self, x, include_first_step=True):
        """
        Simple method to run a specific action only every x steps
        """

        if include_first_step:
            return self.step % x == 0 or self.step == self.steps

        return (self.step != 0 and self.step % x == 0) or self.step == self.steps

    def test_step(self, rank_every_x_steps=2500):
        """
        Test the model using the entire test set, additionally also execute a ranking test
        """

        x, y_p, y_n = self.dataset.test.pairs()
        feed_dict = self.model.feed_dict(x, y_p, y_n)

        if self.every_x_steps(x=rank_every_x_steps):

            # Run our self-designed CompVecEval evaluation metric
            t0 = time.time()
            compveceval_r = self.evaluation.compveceval.evaluate(self.session, self.compose_op, self.compose_y_op)
            feed_dict = self.evaluation.compveceval.feed_dict(compveceval_r, feed_dict=feed_dict)

            # Run word similarity evaluation for metrics as fromwordvectors.org
            t1 = time.time()
            wordsim_r = self.evaluation.wordsim.evaluate()
            feed_dict = self.evaluation.wordsim.feed_dict(wordsim_r, feed_dict=feed_dict)

            # Run evaluation from SentEval to get sentence metrics
            t2 = time.time()
            thorough_eval = self.step == self.steps or (self.step == 0 and self.model.composition in ['sum', 'avg', 'prod', 'max'])
            senteval_tf_r, senteval_r = self.evaluation.senteval.evaluate(self.session, self.compose_e_op, pretrain=self.pretraining, thorough=thorough_eval)
            feed_dict = self.evaluation.senteval.feed_dict(senteval_tf_r, feed_dict=feed_dict)

            t3 = time.time()
            loss, _, summary = self.session.run([
                self.loss_op,
                self.compveceval_op,
                self.summary_loss_evaluation_op
            ], feed_dict=feed_dict)

            for k, v in compveceval_r.items():
                if k == 'MNR':
                    self.out(k, v, format='%.3f')
                else:
                    self.out(k, v, format='%.2f%%')

            for k, v in wordsim_r.items():
                self.out(k, v, format='%.3f')

            for k, v in senteval_r.items():
                self.out(k, v, format='%.3f')

            self.out('CompVecEval Time', (t1 - t0), format='%.2fs')
            self.out('WordSim Time', (t2 - t1), format='%.2fs')
            self.out('SentEval Time', (t3 - t2), format='%.2fs')

        else:
            loss, summary = self.session.run([
                self.loss_op, self.summary_loss_op
            ], feed_dict=feed_dict)

        self.test_writer.add_summary(summary, self.step)

        if math.isnan(loss):
            self.out('Test loss is NaN, stopped training and stop run')
            return False

        self.out('Test Loss', loss)
        return True

    # noinspection PyAttributeOutsideInit
    def train_step(self, out_every_x_steps=100, summary_every_x_steps=100):
        """
        Train the model using a batch of training data
        """

        # Train step
        if self.x_max_buckets is None:
            bucket = None
            x, y_p, y_n = self.dataset.train.next_batch(self.batch_size)
            optimize_op, loss_op, summary_loss_ops = self.optimize_op, self.loss_op, self.summary_loss_op
        else:
            x, y_p, y_n, bucket = self.dataset.train.next_batch(self.batch_size)
            optimize_op, loss_op, summary_loss_ops = self.optimize_ops[bucket], self.loss_ops[bucket], self.summary_loss_ops[bucket]

        feed_dict = self.model.feed_dict(
            x, y_p, y_n,
            dropout_keep_p=self.dropout_keep_p,
            stop_embedding_gradients=(self.model.composition not in ['sum', 'avg', 'prod', 'max'] and self.step < self.refine_after_x_steps),
            bucket=bucket
        )

        # Only execute summary for train every 10 steps to speed up training
        if self.every_x_steps(summary_every_x_steps):
            _, loss, summary = self.session.run([
                optimize_op, loss_op, summary_loss_ops
            ], feed_dict=feed_dict)
            self.train_writer.add_summary(summary, self.step)
        else:
            _, loss = self.session.run([optimize_op, loss_op], feed_dict=feed_dict)

        if math.isnan(loss):
            self.out('Train loss is NaN, stopped training and stop run')
            return False

        self.average_loss += loss

        if self.every_x_steps(x=out_every_x_steps):
            if self.step > 0:
                self.average_loss /= out_every_x_steps

            # The average loss is an estimate of the loss over the last 100 batches.
            self.out('Train Loss', self.average_loss)
            self.average_loss = 0

        return True

    def save_model(self, save_embeddings=True, save_tensorflow_model=True, save_out=True):
        """
        Save the Tensorflow model as well as the embedding data
        """

        if save_embeddings:
            embeddings_matrix = self.model.embeddings.eval()
            embeddings = dict()

            for i in range(self.dataset.vocab_size):
                embeddings[i] = embeddings_matrix[i, :]

            save_path = self.embedding_processor.store_embeddings(embeddings, '%s/embeddings/%s-%d.vec.gz' % (self.run_dir, self.name, self.step))
            self.out("Embedding vectors saved in file: %s" % save_path)

        # Save the variables to disk.
        if save_tensorflow_model:
            save_path = self.saver.save(self.session, '%s/logs/%s/test/%d.ckpt' % (self.run_dir, self.name, self.step))
            self.out("Model saved in file: %s" % save_path)

        # Save the training out to disk
        if save_out:
            save_path = '%s/output/%s-%d.pkl' % (self.run_dir, self.name, self.step)
            pickle.dump(self.store_attr, open(save_path, 'w+b'), protocol=4)
            self.out("Training output saved in file: %s" % save_path)

    def out(self, key, value=None, format='%.4f'):
        """
        Push string or value to print and save later
        """

        if value is None:
            self.out_lines.append(key)
        else:
            self.out_attr[key] = format % value
            self.store_attr[self.step][key] = value

    # noinspection PyAttributeOutsideInit
    def print_out(self):
        """
        Print step results
        """

        elapsed_time = time.time() - self.start_time
        self.start_time = time.time()

        print('[%s] [%d/%d] [epoch %d] [%.2fs]' % (
            self.name, self.step, self.steps, self.dataset.train.epochs_completed, elapsed_time
        ), end=' ')

        out_attr = []

        for key, value in self.out_attr.items():

            if key.endswith('/n') or key.endswith('/pearson/p') or key.endswith('/spearman/r') or key.endswith('/spearman/p'):
                continue

            if key.endswith('/pearson/r'):
                key = key.replace('/pearson/r', '')

            out_attr.append('%s: %s' % (key, value))

        print(', '.join(out_attr), end='\n')

        for value in self.out_lines:
            print(value)

        self.out_attr = dict()
        self.out_lines = list()
