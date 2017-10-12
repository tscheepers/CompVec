# coding: utf-8

import tensorflow as tf
import numpy as np


class Model:
    """
    The model that trains embeddings for composition under the method c_fn()
    """

    def __init__(self, batch_size=None,
                 x_max_length=32,
                 y_max_length=16,
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
        self.x_max_length = x_max_length
        self.y_max_length = y_max_length
        self.summaries = []

        # To enter values from the vocabulary
        self.x = tf.placeholder(tf.int32, shape=(batch_size, x_max_length), name='x')
        self.x_mask = tf.placeholder(tf.int32, shape=(batch_size, x_max_length), name='x_mask')

        self.y_p = tf.placeholder(tf.int32, shape=(batch_size, y_max_length), name='y_positive')
        self.y_p_mask = tf.placeholder(tf.int32, shape=(batch_size, y_max_length), name='y_positive_mask')

        self.y_n = tf.placeholder(tf.int32, shape=(batch_size, y_max_length), name='y_negative')
        self.y_n_mask = tf.placeholder(tf.int32, shape=(batch_size, y_max_length), name='y_negative_mask')

        # To enter embeddings directly in evaluation
        self.x_e = tf.placeholder(tf.float32, shape=(batch_size, x_max_length, embedding_size), name='x_e')

        # To set the dropout paramter
        self.dropout_keep_p = tf.placeholder(tf.float32, name='dropout_keep_p')
        self.stop_embedding_gradients = tf.placeholder(tf.int32, name='stop_embedding_gradients')

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
            xe = tf.nn.embedding_lookup(self.embeddings, self.x)

            # We mask the inputs
            mask = tf.cast(tf.transpose(self.x_mask), tf.float32)
            xe = tf.transpose(tf.transpose(xe) * mask)

            # We apply dropout for regularization
            if dropout:
                xe = tf.nn.dropout(xe, self.dropout_keep_p)

            xe = tf.cond(
                tf.equal(self.stop_embedding_gradients, tf.constant(1)),
                lambda: tf.stop_gradient(xe),
                lambda: xe
            )

        with tf.variable_scope('y_p_embedding_lookup'):
            # Positive target
            ye_p = tf.nn.embedding_lookup(self.embeddings, self.y_p)

            # We mask the inputs
            mask = tf.cast(tf.transpose(self.y_p_mask), tf.float32)
            ye_p = tf.transpose(tf.transpose(ye_p) * mask)

            ye_p = tf.cond(
                tf.equal(self.stop_embedding_gradients, tf.constant(1)),
                lambda: tf.stop_gradient(ye_p),
                lambda: ye_p
            )

        with tf.variable_scope('y_n_embedding_lookup'):
            # Negative target
            ye_n = tf.nn.embedding_lookup(self.embeddings, self.y_n)

            # We mask the inputs
            mask = tf.cast(tf.transpose(self.y_n_mask), tf.float32)
            ye_n = tf.transpose(tf.transpose(ye_n) * mask)

            # During training we do not want to update the negative targets, so we stop the gradients
            if self.stop_gradients_y_n:
                ye_n = tf.stop_gradient(ye_n)
            else:
                ye_n = tf.cond(
                    tf.equal(self.stop_embedding_gradients, tf.constant(1)),
                    lambda: tf.stop_gradient(ye_n),
                    lambda: ye_n
                )

        return xe, ye_p, ye_n

    def c_fn(self, e, mask, reuse=False):
        """
        Compose embeddings e
        """

        with tf.variable_scope('composition'):

            if self.composition == 'sum':
                c = tf.reduce_sum(e, axis=1)
            elif self.composition == 'prod':
                # Problem see: https://github.com/tensorflow/tensorflow/issues/8841
                e = tf.transpose(e) + (1.0 - tf.cast(tf.transpose(mask), tf.float32))
                c = tf.reduce_prod(tf.transpose(e), axis=1)
            elif self.composition == 'max':
                c = tf.reduce_max(e, axis=1)
            elif self.composition == 'gru':
                batch_size = tf.shape(e)[0]

                x_mask = tf.tile(tf.expand_dims(tf.cast(mask, tf.float32), 2), (1, 1, self.embedding_size))

                c = tf.scan(
                    lambda result_prev, x: self.gru_layer(
                        result_prev,  # h_retain_prev
                        x[0],  # e
                        x[1],  # x_mask
                        reuse=reuse
                    ),
                    (tf.transpose(e, [1, 0, 2]), tf.transpose(x_mask, [1, 0, 2])),
                    initializer=tf.zeros([batch_size, self.embedding_size])
                )[self.x_max_length - 1]

            elif self.composition == 'rnn':
                batch_size = tf.shape(e)[0]

                x_mask = tf.tile(tf.expand_dims(tf.cast(mask, tf.float32), 2), (1, 1, self.embedding_size))

                c = tf.scan(
                    lambda result_prev, x: self.rnn_layer(
                        result_prev,  # h_retain_prev
                        x[0],  # e
                        x[1],  # x_mask
                        reuse=reuse
                    ),
                    (tf.transpose(e, [1, 0, 2]), tf.transpose(x_mask, [1, 0, 2])),
                    initializer=tf.zeros([batch_size, self.embedding_size])
                )[self.x_max_length - 1]

            elif self.composition == 'cnn':

                c = self.conv_layer(e, reuse=reuse)

            else:  # avg
                c = tf.transpose(tf.reduce_sum(e, axis=1))
                c /= tf.cast(tf.reduce_sum(mask, axis=1), tf.float32)
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
        xe, ye_p, ye_n = self.es_fn()

        # Composed embedding
        xc = self.c_fn(xe, self.x_mask)
        yc_p = self.c_fn(ye_p, self.y_p_mask, reuse=True)
        yc_n = self.c_fn(ye_n, self.y_n_mask, reuse=True)

        # To compose outside embeddings directly
        x_ec = self.c_fn(self.x_e, self.x_mask, reuse=True)

        # Loss
        loss = self.loss_fn(xc, yc_p, yc_n)
        return loss, xc, x_ec, yc_p

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

    def conv_layer(self, e, filter_width=3, reuse=False):
        """
        Convolutional layer
        See: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
        """

        with tf.variable_scope("conv_pool", reuse=reuse):

            # Add padding for wide convolution
            e = tf.pad(e, [(0, 0), (filter_width - 1, filter_width - 1), (0, 0)], "CONSTANT")

            # Expand dims to make it comparable to multi-channel convolutions
            e = tf.expand_dims(e, -1)

            W = tf.get_variable(name='weight_input', shape=(filter_width, self.embedding_size, 1, self.embedding_size),
                                initializer=tf.random_normal_initializer(stddev=0.1))

            conv = tf.nn.conv2d(e, W, strides=(1, 1, 1, 1), padding="VALID", name="conv")

            b = tf.get_variable(name='bias', shape=(self.embedding_size,), initializer=tf.constant_initializer(0.0))

            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            pooled = tf.nn.max_pool(h, ksize=(1, self.x_max_length + filter_width - 1, 1, 1),
                                    strides=(1, 1, 1, 1), padding="VALID", name="pool")

            c = tf.squeeze(pooled)

        return c

    def feed_dict(self, x=None, y_p=None, y_n=None, dropout_keep_p=1.0, stop_embedding_gradients=False):
        """
        Create feed dictionary for the Tensorflow model
        """

        def mask(x):
            return 0 if x == 0 else 1

        mask = np.vectorize(mask)

        if x is not None:
            if x.ndim == 3:  # Input embeddings directly
                result = {
                    self.x_e: x,
                    self.x_mask: mask(np.sum(x, axis=2))
                }
            else:  # Default: input indices for the embedding matrix
                result = {
                    self.x: x,
                    self.x_mask: mask(x),
                    self.dropout_keep_p: dropout_keep_p,
                    self.stop_embedding_gradients: 1 if stop_embedding_gradients else 0
                }
        else:
            result = {
                self.dropout_keep_p: dropout_keep_p,
                self.stop_embedding_gradients: 1 if stop_embedding_gradients else 0
            }

        if y_p is not None:
            result[self.y_p] = y_p
            result[self.y_p_mask] = mask(y_p)

        if y_n is not None:
            result[self.y_n] = y_n
            result[self.y_n_mask] = mask(y_n)

        return result
