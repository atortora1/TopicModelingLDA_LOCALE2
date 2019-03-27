from pymongo import MongoClient
import pprint

client = MongoClient('192.168.195.129', 27017)

for news in  client.fairvalyou.news.find():
    pprint.pprint(news)