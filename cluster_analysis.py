from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


news_df = pd.DataFrame()
client = MongoClient('192.168.195.129', 27017)
news_collection = client.fairvalyou.news
print('Inizio Prelievo dati da mongo')
news_df = pd.DataFrame(news_collection.find())
gruopby_news = news_df.groupby(['cluster']).size()
#print(gruopby_news.index)
pd.set_option('display.max_columns', 30)
display(gruopby_news)
""" Grafico a barre
x = [cluster.split(",")[0] for cluster in gruopby_news.index]
y = gruopby_news.values
#print(y)
plt.figure(figsize=(20,10))
plt.bar(x,y)
plt.show()
"""
"""Grafico a torta 
numero_cluster=[]
info_cluster = []
for cluster in gruopby_news.index:
    numero_cluster.append(int(cluster.split(",")[0]))
    info_cluster.append(cluster.split(",")[1])
groupby_news_frame = pd.DataFrame({'num_cluster': numero_cluster,"info_cluster": info_cluster, 'count':gruopby_news.values})
#groupby_news_frame = groupby_news_frame.set_index('num_cluster')
#groupby_news_frame = groupby_news_frame.sort_index()
groupby_news_frame = groupby_news_frame.sort_values(by='num_cluster')
print(groupby_news_frame)


x = [cluster.split(",")[0] for cluster in gruopby_news.index]
y = [float(count)/sum(gruopby_news.values) for count in gruopby_news.values]
#print(y)
plt.figure(figsize=(20,10))
plt.bar(x,y)
plt.show()


x = groupby_news_frame['num_cluster']
y = [float(count)/sum(groupby_news_frame['count']) for count in groupby_news_frame['count']]
#print(y)
plt.figure(figsize=(20,10))
plt.bar(x,y)
plt.show()
"""