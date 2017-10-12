#!/usr/bin/env python
# coding: utf-8
import datetime
from dataset.wordnet_dataset import WordnetDataset
from dataset.wikipedia_dataset import WikipediaDataset
from dataset.embedding_processor import EmbeddingProcessor
from run import Run
from utils import store_dataset, filter_pretrained, directory
from collections import defaultdict


def main():
    """Entry point of the program"""

    t = "%i" % datetime.datetime.now().timestamp()

    d = WordnetDataset(x_max_length=32, y_max_length=8)
    ep = EmbeddingProcessor(d.vocabulary)
    store_dataset(d)
    filter_pretrained(ep)
    runs(t + '-wordnet', d, ep)

    d = WikipediaDataset(x_max_length=256, y_max_length=8)
    ep = EmbeddingProcessor(d.vocabulary, path=directory('/data/wikipedia/'))
    store_dataset(d)
    filter_pretrained(ep)
    runs(t + '-wikipedia', d, ep)


def runs(n, d, ep):

    Run(n, d, ep, composition='sum', margin=5, pretraining='fasttext').execute()
    Run(n, d, ep, composition='sum', margin=5, pretraining='word2vec').execute()
    Run(n, d, ep, composition='sum', margin=5, pretraining='glove').execute()
    
    Run(n, d, ep, composition='avg').execute()
    Run(n, d, ep, composition='avg', pretraining='fasttext').execute()
    Run(n, d, ep, composition='avg', pretraining='word2vec').execute()
    Run(n, d, ep, composition='avg', pretraining='glove').execute()
    
    Run(n, d, ep, composition='prod').execute()
    Run(n, d, ep, composition='prod', pretraining='fasttext').execute()
    Run(n, d, ep, composition='prod', pretraining='word2vec').execute()
    Run(n, d, ep, composition='prod', pretraining='glove').execute()
    
    Run(n, d, ep, composition='max').execute()
    Run(n, d, ep, composition='max', pretraining='fasttext').execute()
    Run(n, d, ep, composition='max', pretraining='word2vec').execute()
    Run(n, d, ep, composition='max', pretraining='glove').execute()
    
    Run(n, d, ep, composition='rnn').execute()
    Run(n, d, ep, composition='rnn', pretraining='fasttext', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='rnn', pretraining='word2vec', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='rnn', pretraining='glove', refine_after_x_steps=100000).execute()
    
    Run(n, d, ep, composition='gru').execute()
    Run(n, d, ep, composition='gru', pretraining='fasttext', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='gru', pretraining='word2vec', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='gru', pretraining='glove', refine_after_x_steps=100000).execute()
    
    Run(n, d, ep, composition='cnn').execute()
    Run(n, d, ep, composition='cnn', pretraining='fasttext', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='cnn', pretraining='word2vec', refine_after_x_steps=100000).execute()
    Run(n, d, ep, composition='cnn', pretraining='glove', refine_after_x_steps=100000).execute()
    
    Run(n, d, ep, composition='rnn', pretraining='fasttext', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='rnn', pretraining='word2vec', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='rnn', pretraining='glove', refine_after_x_steps=25000).execute()
    
    Run(n, d, ep, composition='gru', pretraining='fasttext', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='gru', pretraining='word2vec', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='gru', pretraining='glove', refine_after_x_steps=25000).execute()
    
    Run(n, d, ep, composition='cnn', pretraining='fasttext', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='cnn', pretraining='word2vec', refine_after_x_steps=25000).execute()
    Run(n, d, ep, composition='cnn', pretraining='glove', refine_after_x_steps=25000).execute()


if __name__ == '__main__':
    main()
