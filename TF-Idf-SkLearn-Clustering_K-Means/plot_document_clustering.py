"""
=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

    HashingVectorizer does not provide IDF weighting as this is a stateless
    model (the fit method does nothing). When IDF weighting is needed it can
    be added by pipelining its output to a TfidfTransformer instance.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce
dimensionality and discover latent patterns in the data.

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.manifold import MDS
from sklearn.cluster import KMeans, MiniBatchKMeans,SpectralClustering,AgglomerativeClustering
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import logging
from optparse import OptionParser
import sys
from time import time
from pymongo import MongoClient
import matplotlib.pyplot as plt
import random
import pprint
import numpy as np
from support_function import *
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--use-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=20000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")
op.add_option("--use-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
#op.add_option("--use-spectral",
             # dest="use_spectral", default=False,
             # help="Use spectral cluster algorithm")
#op.add_option("--use-agglomerative",
       #       dest="use_agglomerative", default=True,
       #       help="Use agglomerative cluster algorithm")

print(__doc__)
op.print_help()


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

client = MongoClient('192.168.195.129', 27017)
news_collection = client.fairvalyou.news
news_df = pd.DataFrame(news_collection.find())
print(news_df.columns)

document = news_df['news'].replace(to_replace='r^[0-9]*$', value='', regex=True).tolist()
print("%d documents" % len(document))
#print("%d categories" % len(dataset.target_names))
print()

#labels = dataset.target
#true_k = np.unique(labels).shape[0]
num_cluster = 50
print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        print('USO HASHING con Tfidf')
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        print('Uso Hashing senza TfIdf')
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    print('USO solo TFidf')

    vectorizer = TfidfVectorizer(max_df=0.5, max_features=None, analyzer = 'word',
                                 min_df=1, stop_words='english',
                                 use_idf=True)

pipeline = Pipeline([
     ('to_dense', DenseTransformer())
])
document
X = vectorizer.fit_transform(document)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    print("Numero Componenti: ", opts.n_components)
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("fatto in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


# #############################################################################
# Do the actual clustering

if opts.minibatch:
    print("*******MINI BATCH KMEANS***********")
    km = MiniBatchKMeans(n_clusters=num_cluster, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
#elif opts.use_spectral:
    #print("*******SPECTRAL CLUSTERING***********")
    #km = SpectralClustering(n_clusters=num_cluster, affinity='precomputed', n_init=100, assign_labels = 'discretize')
#elif opts.use_agglomerative:
   # print("************AGGLOMERATIVE CLUSTERING*********")
    #km = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
       #     connectivity=None, linkage='ward', memory=None, n_clusters=num_cluster,
         #   pooling_func='deprecated')
else:
    print("*******K-MEANS***********")
    km = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
clusters = km.labels_.tolist()
print("done in %0.3fs" % (time() - t0))
print()
"""
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
"""
print()

cluster_colors = dict()
cluster_name_map = dict()


if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    cluster_name = []
    terms = vectorizer.get_feature_names()
    for i in range(num_cluster):
        print("Cluster %d:" % i, end='')
        cluster_colors[i] = randomcolor()
        name = ' '
        for ind in order_centroids[i, :10]:
            name += ' ' + terms[ind]
            print(' %s' % terms[ind], end='')
        cluster_name.append(name)
        cluster_name_map[i] = name.split(",")
        print()

cluster_predicit = km.predict(X)

# ================ Update columns cluster in database ============================
for index, row in news_df.iterrows():
    client.fairvalyou.news.update({"_id": row['_id']}, {"$set": {"cluster": str(cluster_predicit[index])+','+cluster_name[cluster_predicit[index]]}})
