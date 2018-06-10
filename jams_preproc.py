# coding: utf-8
import pandas as pd
import numpy as np
import pymorphy2 as pm2
from sklearn import preprocessing 
import matplotlib.pyplot as plt

from operator import itemgetter
from itertools import groupby
import collections as clcts
import time
import re
import sys 
import random as rnd

class path:
    def __init__(self):
        self.jams_on_month = 'traffic_jams/jams_on_month.csv'
        self.jams_on_week = 'traffic_jams/jams_on_week.csv'
        self.convertibility = 'traffic_jams/convertibility.csv'       
path = path()

def df_to_csv(path = 'traffic_jams/', name = 'preproc_.csv'):
    df.to_csv(path + name, encoding='UTF-8')

def df_from_csv(path = 'traffic_jams/', name = 'preproc_.csv'):
    return pd.read_csv(path + name, encoding='UTF-8')

def get_week_day_jams(jow):
    week_jams = {
        'пн' : [],
        'вт' : [],
        'ср' : [],
        'чт' : [],
        'пт' : [],
        'сб' : [],
        'вс' : []
    }
    for indx, day in zip(range(0, len(jow), 24), week_jams.keys()):
        week_jams[day] = list(jow[indx:indx+24].point)        
    return week_jams

def get_year_month_jams(joy):
    year_month_jams = {
        2016 : [],
        2017 : [],
        2018 : []
    }
    for indx, year in zip(range(1, len(joy), 12), year_month_jams.keys()):
        year_month_jams[year] = list(joy[indx-1:indx+11].weight)    
    return normalize_year_month_jams(year_month_jams)

def normalize_year_month_jams(year_month_jams):
    norm = preprocessing.normalize( list(year_month_jams.values()), norm='l1' )
    for indx, year in enumerate(year_month_jams.keys()):
        year_month_jams[year] = norm[indx]     
    return year_month_jams

def get_compromise_point(points):
    dsp = 0.333
    return np.mean([int(str(points)[0]), int(str(points)[1])]) + rnd.uniform(-dsp, dsp)

def uniq_jam_point(year, month, day, hour):
    global year_month_jams
    global week_day_jams
    global con
    
    if con.point[hour]:
        season_weight = year_month_jams[year][month-1]
    else:
        dsp = 0.010
        season_weight = rnd.uniform(-dsp, dsp)
        
    week_weight = week_day_jams[day][hour]
    if week_weight > 10:
        week_weight = get_compromise_point(week_weight)
    
    return week_weight * (1 + season_weight)

joy = pd.read_csv(path.jams_on_month, encoding='UTF-8')
jow = pd.read_csv(path.jams_on_week, encoding='UTF-8')
con = pd.read_csv(path.convertibility, encoding='UTF-8')
year_month_jams = get_year_month_jams(joy)
week_day_jams = get_week_day_jams(jow)

def get_jams_table(year_month_jams, week_day_jams):
    df = pd.DataFrame(columns=['year', 'month', 'week_day', 'hour', 'point'])
    for year in year_month_jams.keys():
        for month, _ in enumerate(year_month_jams[year], 1):
            for day in week_day_jams.keys():
                for hour, _ in enumerate(week_day_jams[day]):
                    point = uniq_jam_point(year, month, day, hour)
                    df.loc[df.shape[0]] = (year, month, day, hour, point)
    return df
                

