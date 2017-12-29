# CompVec: word embeddings for better compositionality

This repository contains all code that was used to write the paper: "CompVec: word embeddings for better compositionality". In the paper we present both an in-depth analysis of various word embeddings (Word2Vec, GloVe and fastText) in terms of their compositionality as well as a method to tune them towards better compositionality. We find that training the embeddings to compose improves the performance of these word embeddings overall.

Word embeddings are tuned using a simple neural network architecture with definitions and lemmas from the lexicon. This resulted in better embeddings, not only in terms of their compositionality but overall. Even more importantly, our architecture allows for the embeddings to be composed using simple arithmetic operations, which makes these embeddings specifically suitable for production applications such as web search or data mining.

The model is written using Tensorflow and can compose embeddings using [multiple composition functions](src/model.py#L89):

- Sum (algebraic composition)
- Avg/Mean (algebraic composition)
- Prod (algebraic composition)
- Max (algebraic composition)
- Sum Projection (learning to compose)
- RNN (learning to compose)
- GRU (learning to compose)
- Bi-directional GRU (learning to compose)
- CNN (learning to compose)

In our analysis, we evaluate original as well as tuned embeddings using existing word similarity and sentence embedding evaluation methods. But aside from these evaluation measures used in related we also evaluate using a novel ranking method which uses a dictionary based dataset of lemmas and definitions from WordNet. Dictionary definitions are inherently compositional and this makes them very suitable for such an evaluation method. In contrast to other evaluation methods, ours is not invariant to the magnitude of the embedding vector—which we show is essential for composition. We consider this new evaluation method to be a key contribution.

- [CompVecEval](src/evaluate/nn.py), a ball tree nearest neightbour ranking approach using a held out set from wordnet to deterine the compositional power of the embeddings. This is a method we developed ourselves and is not invariant to the embeddingvector's mangitude. Our method also takes the many-to-many lemma definition relationships in wordnet into account.
- [SentEval](src/evaluate/senteval.py), sentence evaluation against various tasks and dataset.
- [WordSim](src/evaluate/wordsim.py), word vector evaluation against various word similairty dataset.

Papers
---

- **Improving Word Embedding Compositionality using Lexicographic Definitions** _(will be published and presented at [WWW '18](https://www2018.thewebconf.org/))_
- **Improving the Compositionality of Word Embeddings** [Thesis PDF](https://thijs.ai/papers/scheepers-msc-thesis-2017-improving-compositionality-word-embeddings.pdf), [Presentation PDF](https://thijs.ai/papers/scheepers-msc-thesis-presentation.pdf)
- **Analyzing the compositional properties of word embeddings** [Paper PDF](https://thijs.ai/papers/scheepers-gavves-kanoulas-analyzing-compositional-properties.pdf)

Please cite the following paper, if you use this code for your own research:

```
@inproceedings{scheepers2018compositionality,
 author = {Scheepers, Thijs and Kanoulas, Evangelos and Gavves, Efstratios},
 title = {Improving Word Embedding Compositionality using Lexicographic Definitions},
 booktitle = {Proceedings of the 27th International Conference on World Wide Web},
 series = {WWW '18},
 year = {2018},
 location = {Lyon, France},
 publisher = {International World Wide Web Conferences Steering Committee},
 address = {Republic and Canton of Geneva, Switzerland},
} 
```

Results
----

Tuned pretrained embeddings can be downloaded at:

- `fasttext_tuned.vec` [Download](http://blob.thijs.ai/compvec/compvec_wordnet_multi/fasttext_tuned.vec.gz)
- `glove_tuned.vec` [Download](http://blob.thijs.ai/compvec/compvec_wordnet_multi/glove_tuned.vec.gz)
- `paragram_tuned.vec` [Download](http://blob.thijs.ai/compvec/compvec_wordnet_multi/paragram_tuned.vec.gz)
- `word2vec_tuned.vec` [Download](http://blob.thijs.ai/compvec/compvec_wordnet_multi/word2vec_tuned.vec.gz)


License (MIT)
-----

Copyright 2017 Thijs Scheepers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
