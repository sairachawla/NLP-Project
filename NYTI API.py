#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:32:51 2023

@author: ameliasayes
"""


#pip install --upgrade pynytimes 

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
"""
def clean_text(str_in):
    import re
    tmp = re.sub("[^A-Za-z#!']+", " ",str_in).lower().strip()
    return tmp


all_df["abstract_clean"] = all_df["abstract"].apply(clean_text)"""


def rem_sw(var_in):
    from nltk.corpus import stopwords
    sw = stopwords.words("english")
    tmp = var_in.split()
    tmp_ar = [word_t for word_t in tmp if word_t not in sw]
    tmp_o = ' '.join(tmp_ar)
    return tmp_o


all_df["abstract_sw"] = all_df["abstract"].apply(rem_sw)


"""def stem_fun(txt_in):
    from nltk.stem import PorterStemmer
    stem_tmp = PorterStemmer()
    tmp = [stem_tmp.stem(word) for word in txt_in.split()]
    tmp = ' '.join(tmp)
    # tmp = list()
    # for word in txt_in.split():
    #     tmp.append(stem_tmp.stem(word))
    return tmp

all_df["abstract_stem"] = all_df["abstract_sw"].apply(stem_fun)"""




"""SEENTIMENT ANALYSIS"""

"""1. VaderSentiment / NLTK"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
all_df["vader_polarity_raw"] = [analyzer.polarity_scores(row)["compound"] for row in all_df["abstract"]]

#all_df["vader_clean"] = [analyzer.polarity_scores(a)["compound"] for a in all_df["abstract_clean"]]

#all_df["vader_sw"] = [analyzer.polarity_scores(a)["compound"] for a in all_df["abstract_sw"]]

#all_df["vader_stem"] = [analyzer.polarity_scores(a)["compound"] for a in all_df["abstract_stem"]]


"""2. TextBlob Sentiment Analyzer"""

#pip install textblob

from textblob import TextBlob

all_df["textblob_polarity_raw"] =  [TextBlob(row).sentiment.polarity for row in all_df["abstract"]]
all_df["textblob_subjectivity_raw"] = [TextBlob(row).sentiment.subjectivity for row in all_df["abstract"]]


"""3. Third Sentiment Analyzer?"""

#flair, polyglot, pattern, stanza are not compatible with latest version of python



"""Classify as Pos / Neg"""
import numpy as np

#create function to classify
def classification(column):
    conditions = [
        (all_df[column] <= -0.5),
        (all_df[column] > -0.5) & (all_df[column] < 0.5),
        (all_df[column] >= 0.5)
        ]
    
    values = ['neg', 'neutral', 'pos']
    
    all_df[column + "_classification"] = np.select(conditions, values)


#TextBlob Classification Lables
classification('textblob_polarity_raw')


    
"""WORD CLOUD AND COUNT VECTORIZER"""

import sklearn

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()

"""CREATE FUNCTION TO IDENTIFY TOP WORDS

word_freq_dict = dict()
def TopWords(president, label, vectorizer, n=None):
    sub_df = all_df[(all_df["president"] == president) & (all_df["textblob_polarity_raw_classification"] == label)]
    corpus = list(sub_df['abstract_sw'])
    
    cnt_vect = vectorizer.fit(corpus)
    bow = vectorizer.transform(corpus)
    sum_words = bow.sum(axis = 0)
    word_frequency = [(word, sum_words[0, indx]) for word, indx in cnt_vect.vocabulary_.items()]
    word_frequency = sorted(word_frequency, key = lambda x: x[1], reverse=True)
    word_frequency = word_frequency[:n]
    return word_freq_dict"""


"""Pos words for Obama"""
""""TopWords('Obama', 'pos', vec, n=None)
obama_pos_top_words = word_frequency"""

sub_df = all_df[(all_df["president"] == "Obama") & (all_df["textblob_polarity_raw_classification"] == "pos")]
corpus = list(sub_df['abstract_sw'])
    
cnt_vect = vec.fit(corpus)
bow = vec.transform(corpus)
sum_words = bow.sum(axis = 0)
word_frequency = [(word, sum_words[0, indx]) for word, indx in cnt_vect.vocabulary_.items()]
word_frequency = sorted(word_frequency, key = lambda x: x[1], reverse=True)
obama_pos_top = word_frequency[:] #top 20 words only

obama_pos_top_words = list()
for i in obama_pos_top:
    word = i[0]
    print(word)
    obama_pos_top_words += (word)
    
str_obamapos = ' '.join(str(x) for x in obama_pos_top_words)




#wordcloud generation
pip install wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

wordcloud = WordCloud().generate(str_obamapos)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

    
"""Neg words for Obama"""
TopWords('Obama', 'neg', vec, n=None)
obama_neg_top_words = word_frequency

"""Pos words for Trump"""
TopWords('Trump', 'pos', vec, n=None)
trump_pos_top_words = word_frequency

"""Neg words for Trump"""
TopWords('Trump', 'neg', vec, n=None)
trump_neg_top_words = word_frequency



"""MODELLING APPROACH - Saira"""



"""
TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

xform = TfidfVectorizer(ngram_range=(1, 1))
xform_df_obama = pd.DataFrame(xform.fit_transform(obama_df['abstract']).toarray())
xform_df_obama.columns = xform.get_feature_names()

xform_df_trump = pd.DataFrame(xform.fit_transform(trump_df['abstract']).toarray())
xform_df_trump.columns = xform.get_feature_names()"""

