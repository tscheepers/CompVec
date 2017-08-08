# coding: utf-8

import tensorflow as tf
import numpy as np


class Model:
    """
    The model that trains embeddings for composition under the method c_fn()
    """

    def __init__(self, batch_size=None,
                 max_definition_length=32,
                 embedding_size=300,
                 vocab_size=10000,
                 valid_size=10,
                 valid_window_start=25,
                 valid_window_end=100,
                 margin=1.0,
                 stop_gradients_y_n=False,
                 composition='avg'):
        """
        Define model parameters
        """

        self.embedding_size = embedding_size
        self.margin = margin
        self.composition = composition
        self.stop_gradients_y_n = stop_gradients_y_n
        self.max_definition_length = max_definition_length
        self.summaries = []

        # To enter values from the vocabulary
        self.x = tf.placeholder(tf.int32, shape=(batch_size, max_definition_length), name='x')
        self.x_mask = tf.placeholder(tf.int32, shape=(batch_size, max_definition_length), name='x_mask')
        self.y_p = tf.placeholder(tf.int32, shape=(batch_size,), name='y_positive')
        self.y_n = tf.placeholder(tf.int32, shape=(batch_size,), name='y_negative')

        # To enter embeddings directly in evaluation
        self.x_e = tf.placeholder(tf.float32, shape=(batch_size, max_definition_length, embedding_size), name='x_e')

        # To set the dropout paramter
        self.dropout_keep_p = tf.placeholder(tf.float32, name='dropout_keep_p')

        self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='embeddings')

        # Random set of words to evaluate similarity on.
        # Only pick dev samples in the head of the distribution.
        np.random.seed(1)
        self.valid_size = valid_size
        self.valid_examples = np.random.choice(range(valid_window_start, valid_window_end), valid_size, replace=False)
        self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

    def es_fn(self, dropout=True):
        """
        Lookup embedding matrices
        """

        with tf.variable_scope('x_embeddings_lookup'):
            e = tf.nn.embedding_lookup(self.embeddings, self.x)

            # We mask the inputs
            mask = tf.cast(tf.transpose(self.x_mask), tf.float32)
            e = tf.transpose(tf.transpose(e) * mask)

            # We apply dropout for regularization
            if dropout:
                e = tf.nn.dropout(e, self.dropout_keep_p)

        with tf.variable_scope('y_p_embedding_lookup'):
            # Positive target
            ye_p = tf.nn.embedding_lookup(self.embeddings, self.y_p)

        with tf.variable_scope('y_n_embedding_lookup'):
            # Negative target
            ye_n = tf.nn.embedding_lookup(self.embeddings, self.y_n)

            # During training we do not want to update the negative targets, so we stop the gradients
            if self.stop_gradients_y_n:
                ye_n = tf.stop_gradient(ye_n)

        return e, ye_p, ye_n

    def c_fn(self, e, reuse=False):
        """
        Compose embeddings e
        """

        with tf.variable_scope('composition'):

            if self.composition == 'sum':
                c = tf.reduce_sum(e, axis=1)
            # elif self.composition == 'prod':
            #     # TODO: fix masking
            #     # Problem see: https://github.com/tensorflow/tensorflow/issues/8841
            #     c = tf.transpose(tf.reduce_prod(e, axis=1))
            elif self.composition == 'max':
                c = tf.reduce_max(e, axis=1)
            elif self.composition == 'gru':
                batch_size = tf.shape(e)[0]

                x_mask = tf.tile(tf.expand_dims(tf.cast(self.x_mask, tf.float32), 2), (1, 1, self.embedding_size))

                c = tf.scan(
                    lambda result_prev, x: self.gru_layer(
                        result_prev,  # h_retain_prev
                        x[0],  # e
                        x[1],  # x_mask
                        reuse=reuse
                    ),
                    (tf.transpose(e, [1, 0, 2]), tf.transpose(x_mask, [1, 0, 2])),
                    initializer=tf.zeros([batch_size, self.embedding_size])
                )[self.max_definition_length - 1]

            elif self.composition == 'rnn':
                batch_size = tf.shape(e)[0]

                x_mask = tf.tile(tf.expand_dims(tf.cast(self.x_mask, tf.float32), 2), (1, 1, self.embedding_size))

                c = tf.scan(
                    lambda result_prev, x: self.rnn_layer(
                        result_prev,  # h_retain_prev
                        x[0],  # e
                        x[1],  # x_mask
                        reuse=reuse
                    ),
                    (tf.transpose(e, [1, 0, 2]), tf.transpose(x_mask, [1, 0, 2])),
                    initializer=tf.zeros([batch_size, self.embedding_size])
                )[self.max_definition_length - 1]

            else:  # avg
                c = tf.transpose(tf.reduce_sum(e, axis=1))
                c /= tf.cast(tf.reduce_sum(self.x_mask, axis=1), tf.float32)
                c = tf.transpose(c)

        return c

    def loss_fn(self, c, ye_p, ye_n):
        """
        Define a triplet loss function with a positive and negative example
        See: https://stackoverflow.com/questions/38260113/implementing-contrastive-loss-and-triplet-loss-in-tensorflow
        """

        with tf.variable_scope('loss'):

            # Distance
            d_p = tf.reduce_sum(tf.square(c - ye_p), axis=1)
            d_n = tf.reduce_sum(tf.square(c - ye_n), axis=1)
            d = d_p - d_n

            # Triplet loss
            loss = tf.reduce_mean(
                tf.maximum(0., self.margin + d)
            )

        self.summaries = [
            tf.summary.scalar('loss/loss', loss),
            tf.summary.scalar('loss/distance', tf.reduce_mean(d)),
            tf.summary.scalar('loss/positive_distance', tf.reduce_mean(d_p)),
            tf.summary.scalar('loss/negative_distance', tf.reduce_mean(d_n)),
        ]

        return loss

    def fn(self):
        """
        Returns the full model for training
        """

        # Embedded values
        e, ye_p, ye_n = self.es_fn()

        # Composed embedding
        compose_x = self.c_fn(e)
        # To compose outside embeddings directly
        compose_x_e = self.c_fn(self.x_e, reuse=True)

        # Loss
        loss = self.loss_fn(compose_x, ye_p, ye_n)
        return loss, compose_x, compose_x_e

    def similarity(self):
        """
        Computes cosine similarity
        """

        normalized_embeddings = self.normalized_embeddings()
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self.valid_dataset)

        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        return similarity

    def normalized_embeddings(self):
        """
        Computes the normalized embeddings for cosine similarity
        """

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        normalized_embeddings = self.embeddings / norm

        return normalized_embeddings

    def gru_layer(self, h_prev, x, x_mask, name='gru', x_dim=300, y_dim=300, reuse=False):
        """
        GRU layer for gru composition
        """

        with tf.variable_scope(name, reuse=reuse):
            # Reset gate
            with tf.variable_scope('reset_gate'):
                Wi_r = tf.get_variable(name='weight_input', shape=(x_dim, y_dim),
                                       initializer=tf.random_normal_initializer(stddev=0.01))
                Wh_r = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim),
                                       initializer=tf.orthogonal_initializer(0.01))
                b_r = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
                r = tf.sigmoid(tf.matmul(x, Wi_r) + tf.matmul(h_prev, Wh_r) + b_r)

            # Update gate
            with tf.variable_scope('update_gate'):
                Wi_z = tf.get_variable(name='weight_input', shape=(x_dim, y_dim),
                                       initializer=tf.random_normal_initializer(stddev=0.01))
                Wh_z = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim),
                                       initializer=tf.orthogonal_initializer(0.01))
                b_z = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
                z = tf.sigmoid(tf.matmul(x, Wi_z) + tf.matmul(h_prev, Wh_z) + b_z)

            # Candidate update
            with tf.variable_scope('candidate_update'):
                Wi_h_tilde = tf.get_variable(name='weight_input', shape=(x_dim, y_dim),
                                             initializer=tf.random_normal_initializer(stddev=0.01))
                Wh_h_tilde = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim),
                                             initializer=tf.orthogonal_initializer(0.01))
                b_h_tilde = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
                h_tilde = tf.tanh(tf.matmul(x, Wi_h_tilde) + tf.matmul(r * h_prev, Wh_h_tilde) + b_h_tilde)

            # Final update
            h = tf.subtract(np.float32(1.0), z) * h_prev + z * h_tilde

            # Force reset hidden state: is h_prev is retain vector consists of ones,
            # is h if retain vector consists of zeros
            h_retain = tf.subtract(np.float32(1.0), x_mask) * h_prev + x_mask * h

        return h_retain

    def rnn_layer(self, h_prev, x, x_mask, name='rnn', x_dim=300, y_dim=300, reuse=False):
        """
        RNN layer for RNN composition
        """

        with tf.variable_scope(name, reuse=reuse):
            Wi = tf.get_variable(name='weight_input', shape=(x_dim, y_dim),
                                 initializer=tf.random_normal_initializer(stddev=0.01))
            Wh = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim),
                                 initializer=tf.orthogonal_initializer(0.01))
            b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))

            h = tf.tanh(tf.matmul(x, Wi) + tf.matmul(h_prev, Wh) + b)

            # Force reset hidden state: is h_prev is retain vector consists of ones,
            # is h if retain vector consists of zeros
            h_retain = tf.subtract(np.float32(1.0), x_mask) * h_prev + x_mask * h

        return h_retain

    def feed_dict(self, x, y_p=None, y_n=None, dropout_keep_p=1.0):
        """
        Create feed dictionary for the Tensorflow model
        """

        def mask(x):
            return 0 if x == 0 else 1

        mask = np.vectorize(mask)

        if x.ndim == 3:  # Input embeddings directly
            result = {
                self.x_e: x,
                self.x_mask: mask(np.sum(x, axis=2))
            }
        else:  # Default: input indices for the embedding matrix
            result = {
                self.x: x,
                self.x_mask: mask(x),
                self.dropout_keep_p: dropout_keep_p
            }

        if y_p is not None:
            result[self.y_p] = y_p

        if y_n is not None:
            result[self.y_n] = y_n

        return result
