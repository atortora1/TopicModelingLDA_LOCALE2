import math
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import nltk
import functions.support_function as f
from nltk.corpus import wordnet as wn
from mongodb.get_data_from_mongo import read_data_mongo
from spacy.lang.en import English
import random
import pandas as pd
import csv
import gensim
from gensim import corpora
import pyLDAvis.gensim
import pickle
from gensim.corpora import Dictionary

nltk.download('wordnet')
spacy.load('en_core_web_sm')

parser = English()

text_data = []
news_df = read_data_mongo( '192.168.195.129', 27017, 'fairvalyou','news')
print(news_df.dtypes)

text_data = news_df['news'].apply(lambda row: f.prepare_text(row))
for token in text_data:
    print(token)

dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
print(dictionary)

pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')


NUM_TOPICS = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=50)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
    #print(topic)

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
