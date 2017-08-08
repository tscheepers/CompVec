#!/usr/bin/env python
# coding: utf-8
import datetime
import os

import tensorflow as tf
import math

from dataset.dataset import Dataset
from dataset.embedding_processor import EmbeddingProcessor
from evaluate.evaluate import Evaluation
from model import Model
from tensorflow.contrib.tensorboard.plugins import projector

from run import Run
from utils import directory, original_embedding_file


def main():
    """Entry point of the program"""

    # Dataset
    d = Dataset(max_definition_length=32)
    ep = EmbeddingProcessor(d.vocabulary)
    n = "%i" % datetime.datetime.now().timestamp()

    store_dataset(d)
    # filter_pretrained(ped)

    # Run(n, d, ep, composition='sum', dropout_keep_p=0.75, margin=5,).execute()
    # Run(n, d, ep, composition='sum', dropout_keep_p=0.75, margin=5, pretraining='fasttext').execute()
    # Run(n, d, ep, composition='sum', dropout_keep_p=0.75, margin=5, pretraining='word2vec').execute()
    # Run(n, d, ep, composition='sum', dropout_keep_p=0.75, margin=5, pretraining='glove').execute()

    # Run(n, d, ep, composition='avg', dropout_keep_p=0.75, margin=0.25).execute()
    # Run(n, d, ep, composition='avg', dropout_keep_p=0.75, margin=0.25, pretraining='fasttext').execute()
    # Run(n, d, ep, composition='avg', dropout_keep_p=0.75, margin=0.25, pretraining='word2vec').execute()
    # Run(n, d, ep, composition='avg', dropout_keep_p=0.75, margin=0.25, pretraining='glove').execute()

    Run(n, d, ep, composition='rnn', dropout_keep_p=0.75, margin=0.25).execute()
    Run(n, d, ep, composition='rnn', dropout_keep_p=0.75, margin=0.25, pretraining='fasttext').execute()
    Run(n, d, ep, composition='rnn', dropout_keep_p=0.75, margin=0.25, pretraining='word2vec').execute()
    Run(n, d, ep, composition='rnn', dropout_keep_p=0.75, margin=0.25, pretraining='glove').execute()

    Run(n, d, ep, composition='gru', dropout_keep_p=0.75, margin=0.25).execute()
    Run(n, d, ep, composition='gru', dropout_keep_p=0.75, margin=0.25, pretraining='fasttext').execute()
    Run(n, d, ep, composition='gru', dropout_keep_p=0.75, margin=0.25, pretraining='word2vec').execute()
    Run(n, d, ep, composition='gru', dropout_keep_p=0.75, margin=0.25, pretraining='glove').execute()


def store_dataset(d):
    """Store dataset for use in other applications"""
    store_dataset_path = directory('/data/compositional_wordnet')
    d.store_dataset(store_dataset_path)


def filter_pretrained(ped):
    """Filter pretrained embeddings using our vocabulary, and store for future use"""
    store_dataset_path = directory('/data/compositional_wordnet')

    ped.process_pretrained_embeddings(input_filename=original_embedding_file('word2vec'),
                                      output_filename=store_dataset_path + '/word2vec.vec.gz')
    ped.process_pretrained_embeddings(input_filename=original_embedding_file('glove'),
                                      output_filename=store_dataset_path + '/glove.vec.gz')
    ped.process_pretrained_embeddings(input_filename=original_embedding_file('fasttext'),
                                      output_filename=store_dataset_path + '/fasttext.vec.gz')


if __name__ == '__main__':
    main()
