# coding: utf-8
import pandas as pd
import numpy as np
import pymorphy2 as pm2
from sklearn import preprocessing
import matplotlib.pyplot as plt

from operator import itemgetter
from itertools import groupby
import collections as clcts
import random as rnd
import time
import re
import sys 

from math import pi, e

class path:
    def __init__(self):
        self.WEATHER_PREPROC = 'holydays/preproc_1.csv'     
path = path()

def df_to_csv(path = 'holydays/', name = 'preproc_.csv'):
    df.to_csv(path + name, encoding='UTF-8')

def df_from_csv(path = 'holydays/', name = 'preproc_.csv'):
    return pd.read_csv(path + name, encoding='UTF-8')

def dt_to_num(date):
    return int(time.mktime(time.strptime(date, '%d.%m.%Y')))

def num_to_dt(date):
    return time.strftime( '%d.%m.%Y', time.gmtime(date + 3600*5) )
   
def numerization_date_values(series):
    for indx, val in series.items():
        series[indx] = dt_to_num(str(val))
        
def add_rand_to_weights(df):
    rnd_weights = pd.Series([e + rnd.uniform(-1, 1) for e in df.weight]) / 10
    df.weight.update(rnd_weights)

numerization_date_values(df.date)
add_rand_to_weights(df)

