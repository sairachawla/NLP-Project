#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:32:51 2023

@author: ameliasayes
"""


"""DATA IMPORT AND SETUP"""

#pip install --upgrade pynytimes 

import os
os.chdir('/Users/ameliasayes/Documents/QMSS/QMSS - Spring/NLP/NLP Project Data/NLP-Project')

from pynytimes import NYTAPI
import pandas as pd

from config import *

nyt = NYTAPI(
    key=key,
    parse_dates=True,
)


""" NYT API syntax for reference:
articles = nyt.article_search(

    query = "George Floyd protest",

    results = 100,

    dates =date_dict,

    options = options_dict)

"""


"""Search queries"""
#dates limited to first term of presidency only
#final results should be 1k minimum for each presidency
obama_pres = nyt.article_search(query = 'Obama', results = 2100, dates={"begin": date(2009, 1, 20), "end": date(2013, 1, 19)})
                                
trump_pres = nyt.article_search(query = 'Trump', results = 2100, dates={"begin": date(2017, 1, 20), "end": date(2020, 1, 19)})



"""create dataframe"""
obama_abstracts = [article['abstract'] for article in obama_pres]
obama_df = pd.DataFrame(obama_abstracts, columns = ['abstract'])
all_df['president'] = 'Obama'


trump_abstracts = [article['abstract'] for article in trump_pres]
trump_df = pd.DataFrame(trump_abstracts, columns = ['abstract'])
trump_df['president'] = 'Trump'


#Merge dataframes
all_df = pd.concat([obama_df, trump_df], axis = 0)
all_df = all_df.reset_index()


"""
Pre-processing for sentiment analysis removed as per instructions from patrick

def clean_text(str_in):
    import re
    tmp = re.sub("[^A-Za-z#!']+", " ",str_in).lower().strip()
    return tmp


all_df["abstract_clean"] = all_df["abstract"].apply(clean_text)

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
"""



"""SEENTIMENT ANALYSIS"""

"""1. VaderSentiment / NLTK"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
all_df["vader_polarity_raw"] = [analyzer.polarity_scores(row)["compound"] for row in all_df["abstract"]]



"""2. TextBlob Sentiment Analyzer"""

#pip install textblob

from textblob import TextBlob

all_df["textblob_polarity_raw"] =  [TextBlob(row).sentiment.polarity for row in all_df["abstract"]]
#add in subjectivity for texblob as well 

all_df["textblob_subjectivity_raw"] = [TextBlob(row).sentiment.subjectivity for row in all_df["abstract"]]




"""Average Sentiment Score"""
import numpy as np

#text blob polarity
sent_score_textblob_metrics = all_df.groupby("president")['textblob_polarity_raw'].agg([np.mean, np.median, np.std, np.min, np.max])
print("Texblob Polarity Scores")
print(sent_score_textblob_metrics)
#Results: mean for Obama articles is 0.066 vs. 0.050 for Trump, not a significant difference


#text blob subjectivity
subj_score_textblob_metrics = all_df.groupby("president")['textblob_subjectivity_raw'].agg([np.mean, np.median, np.std, np.min, np.max])
print("TextBlob Subjectivity Scores")
print(subj_score_textblob_metrics)


#vader polarity
sent_score_vader_metrics = all_df.groupby("president")['vader_polarity_raw'].agg([np.mean, np.median, np.std, np.min, np.max])
print("Vader Polarity Scores")
print(sent_score_vader_metrics)
#Results: Mean for Obama articles is 12.1 vs. 1.4 for Trump, significant difference



"""
Overall: Huge difference in results from the two methods, wonder why this is? 
In both Obama has a higher sentiment, but this is marginal for textblob and not significant 
Implication: methodology can alter results significantly for sentiment analysis
"""


#bar charts of average result
import matplotlib.pyplot as plt
x1 = all_df.groupby("president")['textblob_polarity_raw'].mean()
x2 = all_df.groupby("president")['textblob_subjectivity_raw'].mean()
x3 = all_df.groupby("president")['vader_polarity_raw'].mean()

x1.plot(kind="bar", title = "textblob polarity sentiment scores")
x2.plot(kind="bar", title = "textblob subjectivity sentiment scores")
x3.plot(kind="bar", title = "vader polarity sentiment scores")



#box plot of distributions
import seaborn as sns

tb_polar_bp = sns.boxplot(x='president', y = 'textblob_polarity_raw', data = all_df)
tb_polar_bp

tb_subj_bp = sns.boxplot(x='president', y = 'textblob_subjectivity_raw', data = all_df)

vs_polar_bp  = sns.boxplot(x='president', y = 'vader_polarity_raw', data = all_df)



"""CLASSIFY ABSTRACT AS POSITIVE OR NEGATIVE FROM SENTIMENT SCORE"""
#create positive and negative classifications for each article abstract as labels for modelling

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
classification('vader_polarity_raw')



    
"""CREATING WORD CLOUDS WITH COUNT VECTORIZER"""

"""
in this code we use count vectorizer to identify the most common words in our four groups: 
    1) Obama + positive sentiment
    2) Obama + negative sentiment
    3) Trump + positive sentiment
    4) Trump + negative sentiment
    
    Method:
        - Preprocess text to remove stop words - we don't want these included in our word clouds
        - Extract all words in abstracts that fit the relevant category
        - Fit count vectorizer and bag ow words to apply numerical code to words and sum counts for each word
        - Create list of dictionary of words and their count
        - Create a list of only the top 20 words 
        - Convert this to a string
        - Graph words in word cloud
"""


"""Preprocessing for Count Vectorizer"""

#remove stop words to prevent Them from being included in count vectorizer count
def rem_sw(var_in):
    from nltk.corpus import stopwords
    sw = stopwords.words("english")
    tmp = var_in.split()
    tmp_ar = [word_t for word_t in tmp if word_t not in sw]
    tmp_o = ' '.join(tmp_ar)
    return tmp_o


all_df["abstract_sw"] = all_df["abstract"].apply(rem_sw)



"""import relevant modules and create function"""
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()

"""CREATE FUNCTION TO IDENTIFY TOP WORDS"""
#pip install wordcloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def TopWords(president, label, vectorizer, n=None):
    
    sub_df = all_df[(all_df["president"] == president) & (all_df["vader_polarity_raw_classification"] == label)]
    corpus = list(sub_df['abstract_sw'])
    
    cnt_vect = vectorizer.fit(corpus)
    bow = vectorizer.transform(corpus)
    sum_words = bow.sum(axis = 0)
    word_frequency = [(word, sum_words[0, indx]) for word, indx in cnt_vect.vocabulary_.items()]
    word_frequency = sorted(word_frequency, key = lambda x: x[1], reverse=True)
    top_words_list_of_dict = word_frequency[:n] #top 20 words only?

    top_words_list = list()
    for i in top_words_list_of_dict:
        word = i[0]
        top_words_list.append(word)
    
    str_top_words = ' '.join(str(x) for x in top_words_list)
    
    return top_words_list_of_dict, str_top_words

    #returning the top words list of dict so we can store the total count for each word
    #returning the string of words to see what the top words are easily




"""Top pos words for Obama"""
obama_pos_top_words_count, obama_pos_top_words = TopWords('Obama', 'pos', vec, n=20)
print(obama_pos_top_words_count) 
print(obama_pos_top_words)

#wordcloud
wordcloud = WordCloud().generate(obama_pos_top_words)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


    
"""Neg words for Obama"""
obama_neg_top_words_count, obama_neg_top_words = TopWords('Obama', 'neg', vec, n=20)
print(obama_neg_top_words_count)
print(obama_neg_top_words)

#wordcloud
wordcloud = WordCloud().generate(obama_neg_top_words)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


"""Pos words for Trump"""
trump_pos_top_words_count, trump_pos_top_words = TopWords('Trump', 'pos', vec, n=20)
print(trump_pos_top_words_count)
print(trump_pos_top_words)

#wordcloud
wordcloud = WordCloud().generate(trump_pos_top_words)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


"""Neg words for Trump"""
trump_neg_top_words_count, trump_neg_top_words = TopWords('Trump', 'neg', vec, n=20)
print(trump_neg_top_words_count)
print(trump_neg_top_words)

#wordcloud
wordcloud = WordCloud().generate(trump_neg_top_words)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

## ANOTHER WORD CLOUD VISUALIZATION COULD BE TO REMOVE THE WORDS THAT APPEAR IN ALL CLASSES AND SEE DIFFERENT WORDS


"""MODELLING APPROACH - Saira to complete"""

"""for saira to run on her local machine"""
os.chdir('/Users/sairachawla/Developer/GR5067/nlp_project/NLP-Project')
obama_df = pd.read_csv('obama_dataframe.csv')
trump_df = pd.read_csv('trump_dataframe.csv')

all_df.columns

"""Add a Column that is presient + classification"""
all_df["classification_for_modelling"] = all_df["president"] + " " + all_df["textblob_polarity_raw_classification"]


"""counts for each category"""
all_df.groupby('classification_for_modelling').size()

"""TFIDF"""

## data preprocessing

## preprcoessing functions to apply
def clean_text(str_in):
    import re
    tmp = re.sub("[^A-Za-z#']+", " ",str_in).lower().strip()
    return tmp

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

def stem_fun(txt_in):
    from nltk.stem import PorterStemmer
    stem_tmp = PorterStemmer()
    tmp = [stem_tmp.stem(word) for word in txt_in.split()]
    tmp = ' '.join(tmp)
    # tmp = list()
    # for word in txt_in.split():
    #     tmp.append(stem_tmp.stem(word))
    return tmp

## apply functions
all_df['abstract_clean'] = all_df['abstract'].apply(clean_text)
all_df['abstract_sw'] = all_df['abstract_clean'].apply(rem_sw)
all_df['abstract_stem'] = all_df['abstract_sw'].apply(stem_fun)

## since the signal is higher for neutral, we are balancing the data 

from sklearn.utils import resample

df_majority = all_df[all_df['classification_for_modelling'] == 'Trump neutral']
df_minority1 = all_df[all_df['classification_for_modelling'] == 'Obama neutral']
df_minority2 = all_df[all_df['classification_for_modelling'] == 'Trump pos']
df_minority3 = all_df[all_df['classification_for_modelling'] == 'Obama pos']
df_minority4 = all_df[all_df['classification_for_modelling'] == 'Trump neg']
df_minority5 = all_df[all_df['classification_for_modelling'] == 'Obama neg']

df_minority1_upsampled = resample(df_minority1,
                                  replace=True,
                                  n_samples = all_df['classification_for_modelling'].value_counts()['Trump neutral'],
                                  random_state=42)
df_minority2_upsampled = resample(df_minority2,
                                  replace=True,
                                  n_samples = all_df['classification_for_modelling'].value_counts()['Trump neutral'],
                                  random_state=42)
df_minority3_upsampled = resample(df_minority3,
                                  replace=True,
                                  n_samples = all_df['classification_for_modelling'].value_counts()['Trump neutral'],
                                  random_state=42)
df_minority4_upsampled = resample(df_minority4,
                                  replace=True,
                                  n_samples = all_df['classification_for_modelling'].value_counts()['Trump neutral'],
                                  random_state=42)
df_minority5_upsampled = resample(df_minority5,
                                  replace=True,
                                  n_samples = all_df['classification_for_modelling'].value_counts()['Trump neutral'],
                                  random_state=42)

all_df = pd.concat([df_majority, df_minority1_upsampled, df_minority2_upsampled, df_minority3_upsampled, df_minority4_upsampled, df_minority5_upsampled])
all_df.reset_index(inplace=True)
all_df.drop(['level_0', 'index', 'Unnamed: 0'], axis=1,inplace=True)

# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(all_df['abstract_stem'], all_df['classification_for_modelling'], test_size=0.2, random_state=42)

# vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

xform = TfidfVectorizer(ngram_range=(1, 1))
xform_train_df = pd.DataFrame(xform.fit_transform(X_train).toarray())
xform_train_df.columns = xform.get_feature_names_out()
xform_test_df = xform.transform(X_test)

# modeling with gridsearchcv
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
# define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# perform grid search cross-validation to find best hyperparameters
grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5)
grid_search.fit(xform_train_df, y_train)

# print best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# use best hyperparameters to train and evaluate model on test set
best_rfc = grid_search.best_estimator_
best_rfc.fit(xform_train_df, y_train)
accuracy = best_rfc.score(xform_test_df, y_test)
predictions = best_rfc.predict(xform_test_df)

# print test set accuracy
print("Test set accuracy: {:.2f}".format(accuracy))
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))


## Saira visualizations
all_df['classification_for_modelling'].hist()

y_test.hist()
pd.Series(list(predictions)).hist(alpha=0.75)
plt.xticks(rotation=45)
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend(['True Class', 'Predicted Class'], bbox_to_anchor=(1.05, 1.0), loc='upper left')

# =============================================================================
# from sklearn.metrics import roc_curve, auc
# probs = best_rfc.predict_proba(xform_test_df)
# preds = probs[:, 1] 
# 
# # calculate false positive rate, true positive rate, and threshold
# fpr, tpr, thresholds = roc_curve(y_test, preds)
# 
# # calculate area under the curve (AUC)
# roc_auc = auc(fpr, tpr)
# 
# =============================================================================


