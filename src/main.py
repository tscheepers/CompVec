#!/usr/bin/env python
# coding: utf-8
import datetime
from dataset.wordnet_dataset import WordnetDataset
from dataset.wikipedia_dataset import WikipediaDataset
from dataset.embedding_processor import EmbeddingProcessor
from run import Run
from utils import filter_pretrained, directory


def main():
    """Entry point of the program"""

    n = "%i" % datetime.datetime.now().timestamp()

    single_wordnet(n)
    multi_wordnet(n)
    wikipedia(n)


def single_wordnet(n, should_filter_pretrained=False):

    d = WordnetDataset(x_max_length=32, y_max_length=1)
    ep = EmbeddingProcessor(d.vocabulary, path=directory('/data/compvec_wordnet_single/'))

    d.store_dataset(path=directory('/data/compvec_wordnet_single/'))

    if should_filter_pretrained:
        filter_pretrained(ep)

    runs(n, d, ep, yc=False, neural=False, random=False)
    runs(n, d, ep, yc=True, neural=False, random=False)


def multi_wordnet(n, should_filter_pretrained=False):
    d = WordnetDataset(test_data_path=directory('/data/compvec_wordnet_single/') + 'test_data.gz', x_max_length=32, y_max_length=6)
    ep = EmbeddingProcessor(d.vocabulary, path=directory('/data/compvec_wordnet_multi/'))

    d.store_dataset(path=directory('/data/compvec_wordnet_multi/'))

    if should_filter_pretrained:
        filter_pretrained(ep)

    runs(n, d, ep, neural=True, random=True)


def wikipedia(n, should_filter_pretrained=False):

    d = WikipediaDataset(x_max_lengths=[33, 62, 118, 504], y_max_length=8)
    ep = EmbeddingProcessor(d.vocabulary, path=directory('/data/compvec_wikipedia/'))
    d.store_dataset(path=directory('/data/compvec_wikipedia/'))

    if should_filter_pretrained:
        filter_pretrained(ep)

    runs(n, d, ep, neural=True, random=False)


def runs(n, d, ep, yc=True, neural=True, random=False):
    """
    Args:
        n: Name
        d: Dataset
        ep: EmbeddingProcessor
        yc: Whether or not the target (y) should be composed or not
    """

    runs_pretraining(n, d, ep, yc, pre='paragram', neural=neural)

    if d.name == 'wikipedia':
        runs_pretraining(n, d, ep, yc, pre='glove', neural=neural)

    runs_pretraining(n, d, ep, yc, pre='fasttext', neural=neural)
    runs_pretraining(n, d, ep, yc, pre='word2vec', neural=neural)

    if random:
        runs_pretraining(n, d, ep, yc)


def runs_pretraining(n, d, ep, yc=True, pre=None, neural=True):

    if d.y_max_length > 1 or yc == False:

        for i in range(10):
            Run(n, d, ep, yc, pre, composition='sum', margin=5, no=i).execute(steps=20000, rank_every_x_steps=20000)

        Run(n, d, ep, yc, pre, composition='avg').execute()
        Run(n, d, ep, yc, pre, composition='max').execute()

        if not (d.name == 'wikipedia' and pre == 'glove'):  # Don't run on glove prod, composed vectors will go to zero because of large max_x
            Run(n, d, ep, yc, pre, composition='prod').execute()

    if neural:
        if pre is None:

            Run(n, d, ep, yc, pre, composition='rnn', refine_after_x_steps=0).execute()
            Run(n, d, ep, yc, pre, composition='gru', refine_after_x_steps=0).execute()
            Run(n, d, ep, yc, pre, composition='cnn', refine_after_x_steps=0).execute()
            Run(n, d, ep, yc, pre, composition='sump', refine_after_x_steps=0).execute()

            if d.y_max_length > 1:
                Run(n, d, ep, yc, pre, composition='bigru', refine_after_x_steps=0).execute()
                Run(n, d, ep, yc, pre, composition='cnncpx', refine_after_x_steps=0).execute()
        else:

            Run(n, d, ep, yc, pre, composition='rnn', refine_after_x_steps=2500).execute()
            Run(n, d, ep, yc, pre, composition='gru', refine_after_x_steps=2500).execute()
            Run(n, d, ep, yc, pre, composition='cnn', refine_after_x_steps=2500).execute()
            Run(n, d, ep, yc, pre, composition='sump', refine_after_x_steps=2500).execute()

            if d.y_max_length > 1:
                Run(n, d, ep, yc, pre, composition='bigru', refine_after_x_steps=2500).execute()
                Run(n, d, ep, yc, pre, composition='cnncpx', refine_after_x_steps=2500).execute()

            Run(n, d, ep, yc, pre, composition='rnn', refine_after_x_steps=20000).execute()
            Run(n, d, ep, yc, pre, composition='gru', refine_after_x_steps=20000).execute()
            Run(n, d, ep, yc, pre, composition='cnn', refine_after_x_steps=20000).execute()
            Run(n, d, ep, yc, pre, composition='sump', refine_after_x_steps=20000).execute()

            if d.y_max_length > 1:
                Run(n, d, ep, yc, pre, composition='bigru', refine_after_x_steps=20000).execute()
                Run(n, d, ep, yc, pre, composition='cnncpx', refine_after_x_steps=20000).execute()


if __name__ == '__main__':
    main()
