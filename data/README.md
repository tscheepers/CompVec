Data folder
====

This folder should contain the following files:

- `GoogleNews-vectors-negative300.bin` from [Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
- `glove.840B.300d.txt` from [GloVe](https://nlp.stanford.edu/projects/glove/)
- `wiki.en.vec` from [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

MySQL server
------

There should also be a MySQL server running with the [WordNet](http://wordnet.princeton.edu) dataset.

Pre-trained vectors
------

- **Word2Vec** Google's pretrained word2vec model [Download](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) (1.5GB)

   It includes word vectors for a vocabulary of 3 million words & phrases and phrases that they trained on roughly 100 billion words from a Google News dataset. The vector length is 300 features.
   Can be converted into `.txt` using `word2vec-bin2txt.py`.

- **GloVe** Pre-trained word vectors. [Website](https://nlp.stanford.edu/projects/glove/)
    - Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
    - Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): [glove.42B.300d.zip](http://nlp.stanford.edu/data/glove.42B.300d.zip)
    - Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)
    - Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): [glove.twitter.27B.zip](http://nlp.stanford.edu/data/glove.twitter.27B.zip)

- **FastText** [Facebook's FastText pretrained embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

    Trained on Wikipedia using fastText. These vectors in dimension 300 were obtained using the skip-gram model described in [Bojanowski et al. (2016)](https://arxiv.org/abs/1607.04606) with default parameters.