Data folder
====

This folder should contain the following files:

- `fasttext_wiki_en.vec` [Download](http://blob.thijs.ai/compvec/fasttext_wiki_en.vec.gz), [Source](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
- `glove_840b.vec` [Download](http://blob.thijs.ai/compvec/glove_840b.vec.gz), [Source](https://nlp.stanford.edu/projects/glove/)
- `paragram_merged_xxl_simlex.vec` [Download](http://blob.thijs.ai/compvec/paragram_merged_xxl_simlex.vec.gz), [Source](http://www.cs.cmu.edu/~jwieting/)
- `word2vec_googlenews.vec` [Download](http://blob.thijs.ai/compvec/word2vec_googlenews.vec.gz), [Source](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

And it should contain the following folders, with their included files.

- `wikipedia_data/*` [Download](http://blob.thijs.ai/compvec/wikipedia_data.tar.gz)
- `senteval_data/*` [Download](http://blob.thijs.ai/compvec/senteval_data.tar.gz)
- `wordsim_data/*` [Download](http://blob.thijs.ai/compvec/wordsim_data.tar.gz)

WordNet dataset MySQL server
------

There should also be a MySQL server running with the [WordNet](http://wordnet.princeton.edu) dataset. The SQL script to seed such a database can be found here: [Download](http://blob.thijs.ai/compvec/wordnet_data.tar.gz).

Dataset used for CompVecEval
------

The test dataset for `compvec_wordnet_single/test_data.gz` can be downloaded here: [Download](http://blob.thijs.ai/compvec/compvec_wordnet_single/test_data.gz), this is a specific selection that one should use for **CompVecEval**.

-  `compvec_wordnet_single/test_data.gz` [Download](http://blob.thijs.ai/compvec/compvec_wordnet_single/test_data.gz)
-  `compvec_wordnet_single/train_data.gz` [Download](http://blob.thijs.ai/compvec/compvec_wordnet_single/train_data.gz)
-  `compvec_wordnet_multi/test_data.gz` [Download](http://blob.thijs.ai/compvec/compvec_wordnet_multi/test_data.gz)
-  `compvec_wordnet_multi/train_data.gz` [Download](http://blob.thijs.ai/compvec/compvec_wordnet_multi/train_data.gz)
-  `wikipedia_data/*` [Download](http://blob.thijs.ai/compvec/wikipedia_data.tar.gz)