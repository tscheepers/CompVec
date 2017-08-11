#!/usr/bin/env python
# coding: utf-8
import datetime
from dataset.dataset import Dataset
from dataset.embedding_processor import EmbeddingProcessor
from run import Run
from utils import store_dataset, filter_pretrained


def main():
    """Entry point of the program"""

    # Dataset
    d = Dataset(max_definition_length=32)
    ep = EmbeddingProcessor(d.vocabulary)
    n = "%i" % datetime.datetime.now().timestamp()

    store_dataset(d)
    # filter_pretrained(ep)

    Run(n, d, ep, composition='sum', margin=5,).execute()
    Run(n, d, ep, composition='sum', margin=5, pretraining='fasttext').execute()
    Run(n, d, ep, composition='sum', margin=5, pretraining='word2vec').execute()
    Run(n, d, ep, composition='sum', margin=5, pretraining='glove').execute()

    Run(n, d, ep, composition='avg').execute()
    Run(n, d, ep, composition='avg', pretraining='fasttext').execute()
    Run(n, d, ep, composition='avg', pretraining='word2vec').execute()
    Run(n, d, ep, composition='avg', pretraining='glove').execute()

    Run(n, d, ep, composition='rnn', pretraining='fasttext', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='rnn', pretraining='word2vec', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='rnn', pretraining='glove', refine_after_x_steps=100000).execute()

    Run(n, d, ep, composition='gru', pretraining='fasttext', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='gru', pretraining='word2vec', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='gru', pretraining='glove', refine_after_x_steps=100000).execute()

    Run(n, d, ep, composition='rnn').execute()
    Run(n, d, ep, composition='rnn', pretraining='fasttext', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='rnn', pretraining='word2vec', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='rnn', pretraining='glove', refine_after_x_steps=25000).execute()

    Run(n, d, ep, composition='gru').execute()
    Run(n, d, ep, composition='gru', pretraining='fasttext', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='gru', pretraining='word2vec', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='gru', pretraining='glove', refine_after_x_steps=25000).execute()

    Run(n, d, ep, composition='prod').execute()
    Run(n, d, ep, composition='prod', pretraining='fasttext').execute()
    Run(n, d, ep, composition='prod', pretraining='word2vec').execute()
    Run(n, d, ep, composition='prod', pretraining='glove').execute()

    Run(n, d, ep, composition='max').execute()
    Run(n, d, ep, composition='max', pretraining='fasttext').execute()
    Run(n, d, ep, composition='max', pretraining='word2vec').execute()
    Run(n, d, ep, composition='max', pretraining='glove').execute()


if __name__ == '__main__':
    main()
