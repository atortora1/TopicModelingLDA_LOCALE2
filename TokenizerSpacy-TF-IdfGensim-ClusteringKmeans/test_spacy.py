import spacy
from spacy.lang.en import English
from pymongo import MongoClient
import pandas as pd
from IPython.display import display
parser = English()
import numpy as np
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(u"This is a sentence.")

def extract_currency_relations(doc):
    # merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    for span in spans:
        span.merge()

    relations = []
    for money in filter(lambda w: w.ent_type_ == "MONEY", doc):
        if money.dep_ in ("attr", "dobj"):
            subject = [w for w in money.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                relations.append((subject, money))
        elif money.dep_ == "pobj" and money.head.dep_ == "prep":
            relations.append((money.head.head, money))
    return relations

print('Inizio lettura dati da Mongo')
client = MongoClient('192.168.195.129', 27017)
news_collection = client.fairvalyou.news
news_df = pd.DataFrame(news_collection.find())
display(news_df.columns)
print('Fine lettura dati da Mongo')
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
TEXT = []
doc = nlp(text)
print("DOC TYPE:"+str(type(doc)))
# Analyze syntax
testo = news_df['news'].iloc[1:2].values
news_df['spacy'] = news_df['news'].iloc[:10].map(lambda row: nlp(row))
print(np.array2string(testo))
#àprint(type(news_df['spacy'].values))
#news_df= news_df.dropna()
#display(news_df['spacy'])
doc = nlp(np.array2string(testo))
#news_df['frasi_nominative'] = [chunk.text for chunk in news_df['spacy'].values.noun_chunks]
#display(news_df['frasi_nominative'])
print("DOC TYPE:"+str(type(doc)))
"""
news_df["verbi"]= [token.lemma_ for token in doc if token.pos_ == "VERB"]

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)
"""
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

displacy.serve(doc, style="ent")