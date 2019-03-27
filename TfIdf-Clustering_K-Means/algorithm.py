from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
import email
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from sklearn.manifold import MDS
from pymongo import MongoClient
from support_function import doubledecode
global map_check_token
global map_check_stem

map_check_token = dict()
map_check_stem = dict()

num_clusters = 7


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def tokenize_and_stem(text):
    # global  map_check_stem
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if not (re.search('[=/]', token)):
                #        if (not(token in map_check_stem.keys())):
                #             map_check_token[token] = 1
                filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    #  global  map_check_token
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if not (re.search('[=/]', token)):
                #   if (not(token in map_check_token.keys())):
                #      map_check_token[token] = 1
                filtered_tokens.append(token)
    return filtered_tokens


# Input

client = MongoClient('192.168.195.129', 27017)
news_collection = client.fairvalyou.news
news_df = pd.DataFrame(news_collection.find())
print(news_df.columns)

# Initial the contents of emails into vectors

word_map = dict()

# ================  read the news from data frame exctract from mongo ============================
document = news_df['news'].tolist()
print("%d documents" % len(document))
# ================  turn the document into vector and TFiDF Matrix ============================


stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

# not super pythonic, no, not at all.
# use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
totalvocab_stemmed_tmp = []
totalvocab_tokenized_tmp = []

# print(email_Content[0][:200])

for news in document:
    # print(title)
    allwords_stemmed = tokenize_and_stem(news)  # for each item in 'email_Content', tokenize/stem
    totalvocab_stemmed_tmp.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(news)
    totalvocab_tokenized_tmp.extend(allwords_tokenized)

for (To, St) in zip(totalvocab_tokenized_tmp, totalvocab_stemmed_tmp):  # remove repeated samples
    if (not (To in map_check_token.keys())):
        map_check_token[To] = 1
        totalvocab_stemmed.append(St)
        totalvocab_tokenized.append(To)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

# define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=200000,
                                   min_df=0.1, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(document)  # fit the vectorizer to email_Content
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)

# ================  begin to cluster the emails ============================


km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# joblib.dump(km,  'doc_cluster.pkl')
clusters = km.labels_.tolist()
# print(clusters)

# ================  print out the result ============================
print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

cluster_colors = dict()
cluster_name_map = dict()
cluster_cnt = 0

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    cluster_colors[i] = randomcolor()

    cluster_name = ""
    for ind in order_centroids[i, :8]:  # replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        cluster_name += str(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0]) + ", "

    cluster_name_map[cluster_cnt] = cluster_name

    print()  # add whitespace
    print()  # add whitespace

    print("Cluster %d titles:" % i, end='\n')
    cnt = 0
    print()  # add whitespace
    print()  # add whitespace

    cluster_cnt = cluster_cnt + 1

print()
print()

print(cluster_name_map)

# ================  plot the result in two dimension ============================

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))

# group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

# add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_name_map[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params( \
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

plt.show()  # show the plot
