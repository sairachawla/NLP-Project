#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:32:51 2023

@author: ameliasayes
"""

pip install --upgrade pynytimes 

from datetime import date, datetime
from pynytimes import NYTAPI

from config import *

# Make sure to set parse dates to True so that the dates
# are parsed into datetime.datetime or datetime.date objects
nyt = NYTAPI(
    key=key,
    parse_dates=True,
)

# Search articles about President Biden
#biden = nyt.article_search("biden")

# You can optionally define the dates between which you want the articles to be
#biden_january = nyt.article_search(
#    query="Obama", dates={"begin": date(2021, 1, 1), "end": date(2021, 1, 31)}
#)

# Optionally you can also define
#biden = nyt.article_search(
#    "biden",
#)

# You can optionally define the dates between which you want the articles to be
obama_pres = nyt.article_search(dates={"begin": date(2009, 1, 20), "end": date(2017, 1, 20)}
                                
trump_pres = nyt.article_search(dates={"begin": date(2017, 1, 21), "end": date(2020, 1, 20)}