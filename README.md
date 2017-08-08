# CompVec: A method to tune word embeddings for better compositionality

This repository contains all code that was used to write the paper: "CompVec: A method to tune word embeddings for better compositionality". Which uses the compositional structure available in a lexicon to tune word embeddings for better compositionality.

The respository contains code to tune word embeddings from three sources: Word2Vec (Google News), GloVe and fastText (Wikipedia).

The model is written using Tensorflow and can compose embeddings using multiple methods:

- Sum (algebraic composition)
- Avg/Mean (algebraic composition)
- Prod (algebraic composition)
- Max (algebraic composition)
- RNN (learning to compose)
- GRU (learning to compose)

We train the model on lemma-definition pairs from the WordNet dataset.

We evaluate the model using three collections of methods.

- SentEval, sentence evaluation against various tasks and dataset.
- WordSim, word vector evaluation against various word similairty dataset
- CompVec NN, a ball tree nearest neightbour ranking approach using a held out set from wordnet to deterine the compositional power of the embeddings. This is a method we developed ourselves and is not invariant to the embeddingvector's mangitude. Our method also takes the many-to-many lemma definition relationships in wordnet into account.

License (MIT)
-----

Copyright 2017 Thijs Scheepers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.