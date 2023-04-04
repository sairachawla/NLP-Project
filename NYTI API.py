#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:32:51 2023

@author: ameliasayes
"""


pip install --upgrade pynytimes 

from datetime import date, datetime
from pynytimes import NYTAPI
import pandas as pd

from config import *

key = 'M8zMmwpJDM3RUq2NdehtkyOjwMxHpZoc'

nyt = NYTAPI(
    key=key,
    parse_dates=True,
)


""" syntax 
articles = nyt.article_search(

    query = "George Floyd protest",

    results = 100,

    dates =date_dict,

    options = options_dict)

"""


"""Search queries"""
obama_pres = nyt.article_search(query = 'Obama', dates={"begin": date(2009, 1, 20), "end": date(2017, 1, 20)})
                                
trump_pres = nyt.article_search(query = 'Trump', dates={"begin": date(2017, 1, 21), "end": date(2020, 1, 20)})


"""create dataframe"""
obama_abstracts = [article['abstract'] for article in obama_pres]
obama_df = pd.DataFrame(obama_abstracts, columns = ['abstract'])
obama_df['president'] = 'Obama'



trump_abstracts = [article['abstract'] for article in trump_pres]
trump_df = pd.DataFrame(trump_abstracts, columns = ['abstract'])
trump_df['president'] = 'Trump'


all_df = pd.concat([obama_df, trump_df], axis = 0)



"""Pre-processing"""
def clean_text(str_in):
    import re
    tmp = re.sub("[^A-Za-z#!']+", " ",str_in).lower().strip()
    return tmp


all_df["abstract_clean"] = all_df["abstract"].apply(clean_text)


def rem_sw(var_in):
    from nltk.corpus import stopwords
    sw = stopwords.words("english")
    tmp = var_in.split()
    # tmp_ar = list()
    # for word_t in tmp:
    #     if word_t not in sw:
    #         tmp_ar.append(word_t)
    tmp_ar = [word_t for word_t in tmp if word_t not in sw]
    tmp_o = ' '.join(tmp_ar)
    return tmp_o


all_df["abstract_sw"] = all_df["abstract_clean"].apply(rem_sw)


def stem_fun(txt_in):
    from nltk.stem import PorterStemmer
    stem_tmp = PorterStemmer()
    tmp = [stem_tmp.stem(word) for word in txt_in.split()]
    tmp = ' '.join(tmp)
    # tmp = list()
    # for word in txt_in.split():
    #     tmp.append(stem_tmp.stem(word))
    return tmp

all_df["abstract_stem"] = all_df["abstract_sw"].apply(stem_fun)





"""Sentiment Analysis"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
all_df["vader1"] = [analyzer.polarity_scores(a)["compound"] for a in all_df["abstract"]]

all_df["vader_clean"] = [analyzer.polarity_scores(a)["compound"] for a in all_df["abstract_clean"]]

all_df["vader_sw"] = [analyzer.polarity_scores(a)["compound"] for a in all_df["abstract_sw"]]

all_df["vader_stem"] = [analyzer.polarity_scores(a)["compound"] for a in all_df["abstract_stem"]]




"""
TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

xform = TfidfVectorizer(ngram_range=(1, 1))
xform_df_obama = pd.DataFrame(xform.fit_transform(obama_df['abstract']).toarray())
xform_df_obama.columns = xform.get_feature_names()

xform_df_trump = pd.DataFrame(xform.fit_transform(trump_df['abstract']).toarray())
xform_df_trump.columns = xform.get_feature_names()"""

