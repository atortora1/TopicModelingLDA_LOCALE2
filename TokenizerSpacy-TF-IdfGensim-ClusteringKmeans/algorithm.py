import os
import gensim.downloader as api
from gensim.models import TfidfModel
import gensim.corpora as corpora
from mongodb.get_data_from_mongo import read_data_mongo
import functions.support_function as f
import pandas as pd
from IPython.display import display
from time import time
from multiprocessing import Process, Queue
import math
from sklearn.cluster import KMeans, MiniBatchKMeans
from gensim.sklearn_api import TfIdfTransformer
import pickle


def tokenize_worker(news_df, result_tokenize):
    print("Inizio ",os.getpid())
    #all_news['news'] = all_news['news'].map(lambda row: prepare_text_for_lda(row))
    tok = news_df['news'].map(lambda row: f.prepare_text(row))
    result_tokenize.put(tok.values)
    print("Fine ",os.getpid())
    return


if __name__ == '__main__':

    result_tokenize = Queue()

    news_df = read_data_mongo( '192.168.195.129', 27017, 'fairvalyou','news')

    print()

    print('Inizio tokening Spacy')
    t0 = time()
    num_proc = 2
    #dim_dataframe = int(len(news_df) / num_proc)
    dim_dataframe = 2
    processes = []
    for i in range(0, num_proc):
        p = Process(target=tokenize_worker,
                    args=(news_df.iloc[int(i * dim_dataframe):int((i + 1) * dim_dataframe - 1)], result_tokenize))
        p.start()
        processes.append(p)
    dataset = []
    res = [result_tokenize.get() for p in processes]
    print()
    print('Fine tokening in %fs' % (time() - t0))

    t2 = time()
    for list in res:
        if list is not None:
            for element in list:
                if element is not None:
                    # print element
                    dataset.append(element)

    dictionary = corpora.Dictionary(dataset)
    corpus = [dictionary.doc2bow(text) for text in dataset]
    print(dictionary)
    print(corpus)

    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    model = TfidfModel(corpus)
    vector = model[corpus]
    dataset = [doc for doc in vector]


    #matrix = model.fit_transform(news_df['news'].to_list())# fit model
    #print(matrix)

    """
    dct = Dictionary(dataset)  # fit dictionary
    corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
    model = TfIdfTransformer(dictionary=dct)
    tfidf_corpus = model.fit_transform(corpus)
    #model = TfidfModel(corpus)  # fit model
    #vector = model[corpus[0]]  # apply model to the first corpus document
    """

    num_cluster = 50
    km = MiniBatchKMeans(n_clusters=num_cluster, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)

    km.fit(dataset)
    clusters = km.labels_.tolist()
    print("done in %0.3fs" % (time() - t2))

    print("Top terms per cluster:")


    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    cluster_name = []
    terms = model.get_feature_names()
    for i in range(num_cluster):
        print("Cluster %d:" % i, end='')
        name = ' '
        for ind in order_centroids[i, :10]:
            name += ' ' + terms[ind]
            print(' %s' % terms[ind], end='')
        cluster_name.append(name)
        print()

    cluster_predicit = km.predict(dataset)
