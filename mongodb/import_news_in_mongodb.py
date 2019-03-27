from pymongo import MongoClient
import json
import pandas as pd
from support_function import doubledecode

client = MongoClient('192.168.195.129', 27017)
print("Load news")
all_news = pd.DataFrame()
economy_news = pd.read_excel('dataset/economy-news.xlsx', na_values='null')
economy_news = economy_news.assign(category='economy')
stock_news = pd.read_excel("dataset/stock_market_news.xlsx")
stock_news = stock_news.assign(category='stock')
commodities_news = pd.read_excel("dataset/commodities_news.xlsx")
commodities_news = commodities_news.assign(category='commodities')
technology_news = pd.read_excel("dataset/technology_news.xlsx")
technology_news = technology_news.assign(category='technology')
frames = [economy_news, stock_news, commodities_news, technology_news]
all_news = pd.concat(frames)
all_news = all_news.dropna()
all_news['news'] = all_news['news'].map(lambda x: doubledecode(x,False))
all_news['date'] = all_news['date'].map(lambda row: pd.to_datetime(doubledecode(row, True), format='%b %d, %Y', errors='coerce'))
all_news = all_news.dropna()
all_news = all_news.sort_values(by='date', ascending=False)
print("News loaded")
print("Store news in MongoDB")
deleted_news = client.fairvalyou.news.delete_many({})
print(deleted_news.deleted_count, "news deleted")
client.fairvalyou.news.insert_many(all_news.to_dict(orient='records'))
print("News stored in MongoDB")