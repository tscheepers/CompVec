# coding: utf-8

from evaluate.compveceval import CompVecEvalEvaluation
from evaluate.senteval import SentEvalEvaluation
from evaluate.wordsim import WordSimEvaluation
from dataset.wordnet_data import WordnetData
from utils import directory


class Evaluation:
    """
    Use this class for evaluating the model
    """

    def __init__(self, model, dataset):
        self.m = model
        self.d = dataset

        path = directory('/data/wordnetsingle/') + 'test_data.gz'
        self.compveceval_test = WordnetData.from_path(path, self.d.vocabulary, self.d.x_max_length, self.d.y_max_length)

        self.compveceval = CompVecEvalEvaluation(self.m, self.compveceval_test)
        self.senteval = SentEvalEvaluation(self.m, self.d)
        self.wordsim = WordSimEvaluation(self.m, self.d)

    def log_similarity(self, top_k=8):
        """
        Print a sample from the top items and their nearest neighbors using CosineSim
        """

        sim = self.m.similarity().eval()
        log_str = ''

        for i in range(self.m.valid_size):
            valid_word = self.d.reversed_vocabulary[self.m.valid_examples[i]]
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]

            log_str = '%sNearest to %s:' % (log_str, valid_word)
            for k in range(top_k):
                close_word = self.d.reversed_vocabulary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)

            log_str = '%s\n' % (log_str,)

        return log_str

    def plot_with_labels(self, filename='tsne.png', skip=2, plot_only=250):
        """
        Plot a tSNE graph or the n top values. We skip UNK and PAD.
        """

        try:
            # pylint: disable=g-import-not-at-top
            import matplotlib
            matplotlib.use('Agg')

            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

            embeddings = self.m.normalized_embeddings().eval()
            low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
            labels = [self.d.reversed_vocabulary[i] for i in range(skip, plot_only)]

            assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
            plt.figure(figsize=(18, 18))  # in inches
            for i, label in enumerate(labels, skip):
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                plt.annotate(label,
                             xy=(x, y),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom')

            plt.savefig(filename)
            plt.close()

            return filename

        except ImportError:
            print('Please install sklearn, matplotlib, and scipy to show embeddings.')