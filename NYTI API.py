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

key = key

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



"""TFIDF"""
from sklearn.feature_extraction.text import TfidfVectorizer

xform = TfidfVectorizer(ngram_range=(1, 1))
xform_df_obama = pd.DataFrame(xform.fit_transform(obama_df['abstract']).toarray())
xform_df_obama.columns = xform.get_feature_names()

xform_df_trump = pd.DataFrame(xform.fit_transform(trump_df['abstract']).toarray())
xform_df_trump.columns = xform.get_feature_names()

