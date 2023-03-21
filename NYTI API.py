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

""" Search for articles within presency dates"""
obama_pres = nyt.article_search(results = 200, dates={"begin": date(2009, 1, 20), "end": date(2017, 1, 20)})
                                
trump_pres = nyt.article_search(query = 'Trump', dates={"begin": date(2017, 1, 21), "end": date(2020, 1, 20)})
