# coding: utf-8
import json
import sys

import pickle
import pymysql.cursors
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.porter import *
import hashlib

# Number of words in the wordnet dictionary
# Also all ids from 1 to NUMBER_OF_WORDS_IN_DICT are present in the database
# We use this attribute to deterministically select x words randomly (with seed)
NUMBER_OF_WORDS = 147306
NUMBER_OF_DEFINITIONS = 117659
NUMBER_OF_SENSES = 206941

# Database connection information
DATABASE_HOST = 'localhost'
DATABASE_USER = 'root'
DATABASE_PASS = ''
DATABASE_DB = 'wordnet30'


class WordnetFetcher:
    """
    Creates a nice dataset from lemmas and their definitions
    We also store definitions in pickle files so we do not
    need to refetch every time
    """

    def mysql_connection(self):
        """
        Creates a MySQL database connection
        """
        return pymysql.connect(host=DATABASE_HOST,
                               user=DATABASE_USER,
                               password=DATABASE_PASS,
                               db=DATABASE_DB,
                               charset='utf8mb4',
                               cursorclass=pymysql.cursors.SSCursor)

    def fetch_definitions(self, stem=False, remove_stop_words=False, definition_limit=None):

        pickle_name = hashlib.md5(
            json.dumps(
                ('fetch_definitions', stem, remove_stop_words, definition_limit, __file__),
                sort_keys=True
            ).encode('utf-8')
        ).hexdigest()

        pickle_file = os.path.dirname(os.path.realpath(__file__)) + '/../../out/%s.pkl' % (pickle_name,)

        if os.path.isfile(pickle_file):
            print('Loading %s…' % (pickle_file,))
            return pickle.load(open(pickle_file, 'r+b'))

        connection = self.mysql_connection()

        definitions = dict()
        definition_vocab = defaultdict(int)

        stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')
        punctiation_words = ['.', ',', ';', ':', '(', ')', '`']
        punctiation_chars = ['`']

        try:
            with connection.cursor() as cursor:

                sql = "SELECT `synset`.`synsetid`, `synset`.`definition` FROM `synset` " \
                      "WHERE EXISTS " \
                      "(SELECT 1 " \
                      "FROM `sense` " \
                      "WHERE `sense`.`synsetid` = `synset`.`synsetid`)"

                # Read records
                cursor.execute(sql)

                for i, (identifier, definition) in enumerate(cursor):

                    if definition_limit is not None and i > definition_limit:
                        break

                    if i % 1000 == 0:
                        sys.stdout.write("\rReading definitions… %6.2f%%" % ((100 * i) / float(NUMBER_OF_DEFINITIONS),))

                    tokens = word_tokenize(definition)

                    if remove_stop_words:
                        tokens = [t for t in tokens if t not in stopwords_english]

                    # remove punctuation
                    tokens = [t for t in tokens if t not in punctiation_words]

                    # remove punctuation chars
                    tokens = [''.join(c for c in t if c not in punctiation_chars) for t in tokens]

                    # to lower case
                    tokens = [t.lower() for t in tokens]

                    if stem:
                        tokens = [stemmer.stem(t) for t in tokens]

                    if len(tokens) > 0:
                        definitions[int(identifier)] = tokens

                    for token in tokens:
                        definition_vocab[token] += 1

        finally:
            connection.close()

        print("\rDefinitions read")

        result = (definitions, definition_vocab)

        print('Storing %s' % (pickle_file,))
        pickle.dump(result, open(pickle_file, 'w+b'), protocol=4)

        return result

    def fetch_lemmas(self, definitions, stem=False, multi_word_lemmas=False):

        pickle_name = hashlib.md5(
            json.dumps(
                ('fetch_lemma', definitions, stem, multi_word_lemmas, __file__),
                sort_keys=True
            ).encode('utf-8')
        ).hexdigest()

        pickle_file = os.path.dirname(os.path.realpath(__file__)) + '/../../out/%s.pkl' % (pickle_name,)

        if os.path.isfile(pickle_file):
            print('Loading %s…' % (pickle_file,))
            return pickle.load(open(pickle_file, 'r+b'))

        connection = self.mysql_connection()

        stemmer = PorterStemmer()

        lemma_per_definition = defaultdict(list)
        lemma_vocab = defaultdict(int)

        try:
            with connection.cursor() as cursor:
                sql = "SELECT `sense`.`synsetid`, `word`.`lemma` FROM `sense` " \
                      "INNER JOIN `word` ON `sense`.`wordid` = `word`.`wordid` " \
                      "WHERE `sense`.`synsetid` IN (" + ','.join([str(x) for x in definitions.keys()]) + ")"
                cursor.execute(sql)

                for i, (identifier, lemma) in enumerate(cursor):

                    if i % 1000 == 0:
                        sys.stdout.write("\rReading words… %6.2f%%" % ((100 * i) / float(NUMBER_OF_SENSES),))

                    tokens = word_tokenize(lemma)

                    if stem:
                        tokens = [stemmer.stem(t) for t in tokens]

                    # to lower case
                    tokens = [t.lower() for t in tokens]

                    if len(tokens) > 0:

                        if multi_word_lemmas or len(tokens) == 1:
                            lemma_per_definition[int(identifier)].append(tokens)

                    if multi_word_lemmas or len(tokens) == 1:
                        for token in tokens:
                            lemma_vocab[token] += 1

        finally:
            connection.close()

        print("\rWords read")

        result = (lemma_per_definition, lemma_vocab)

        print('Storing %s' % (pickle_file,))
        pickle.dump(result, open(pickle_file, 'w+b'), protocol=4)

        return result