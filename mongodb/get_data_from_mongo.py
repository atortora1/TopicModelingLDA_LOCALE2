from pymongo import MongoClient
import pandas as pd


def read_data_mongo(ip_client, port, db, collection):
    print('Inizio lettura dati da Mongo')
    client = MongoClient(ip_client, port)
    news_collection = client[db][collection]
    news_df = pd.DataFrame(news_collection.find())
    print('Fine lettura dati da Mongo')
    return news_df

