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

from math import fabs
from collections import Counter

class path:
    def __init__(self):
        self.WEATHER = 'weather/preproc_14.csv'
        self.JAMS = 'traffic_jams/preproc_2.csv'
        self.HOLYDAYS = 'holydays/preproc_3.csv'
        self.CRASH = 'crash/preproc_6.csv'       
path = path()

def get_time_str(date):
    return time.strftime( '%d.%m.%Y %H:%M', time.gmtime(date + 3600*5) )

weath_df = pd.read_csv(path.WEATHER, encoding='UTF-8')
jams_df = pd.read_csv(path.JAMS, encoding='UTF-8')
hol_df = pd.read_csv(path.HOLYDAYS, encoding='UTF-8')
crash_df = pd.read_csv(path.CRASH, encoding='UTF-8')

# collation = pd.DataFrame(columns=['time_', 'lat', 'lon', 'crash_effect', 'holyday_weight', 'jam_point', 
#                                   'temp', 'soil_temp', 'pressure', 'pressure_change', 'humidity', 'wind_speed',
#                                   'cloudy', 'clouds_height', 'visibility', 'precipitations', 'snow', 'soil', 'weath_descr'])

corr_matrix = collation.corr()
np.fill_diagonal(corr_matrix.values, 0)
corr_matrix.max(axis=0)

def get_stick_date_stamp(time_stamp):
    return time.strftime( '%d%m%Y', time.gmtime(time_stamp) )

def get_stick_date_series(time_series):
    return list( map(get_stick_date_stamp, time_series) )

def get_crash_counter(date_series):
    counter_ = Counter()
    for date_ in date_series:
        counter_[date_] += 1        
    return counter_

def merge_crash_counter():
    global collation
    counter_ = get_crash_counter(get_stick_date_series(collation.time_))     
    collation['crash_counter'] = list(
        map(
            lambda time_: counter_[get_stick_date_stamp(time_)], 
            collation.time_ 
        )
    )
    return collation

def add_total_crash_point():
    global collation
    collation['total_crash_point'] = np.round(collation.crash_effect * collation.crash_counter, decimals=0)

add_total_crash_point()

def scatter_plot_(X = None, Y = None, size = (10, 10)):
    plt.grid(True, alpha = 0.2) 
    plt.scatter(X, Y, color='g', alpha = 0.7)
    plt.plot(X, Y, alpha = 0.3)
    plt.rcParams['figure.figsize'] = (size[0], size[1])
    plt.show()

scatter_plot_(range( 0, len(norm) ), norm, size)

size = (27, 10)
scatter_plot_(range( 0, len(collation.crash_counter) ), collation.crash_counter, size)
# collation.crash_counter[collation.crash_counter > 20] = 20

scatter_plot_(range( 0, len(collation.holyday_weight) ), collation.holyday_weight, size)

scatter_plot_(range( 0, len(collation.crash_effect) ), collation.crash_effect, size)
# collation.crash_effect[collation.crash_effect > 10] = 10

scatter_plot_(range( 0, len(collation.jam_point) ), collation.jam_point, size)

scatter_plot_(range( 0, len(collation.temp) ), collation.temp, size)

scatter_plot_(range( 0, len(collation.visibility) ), collation.visibility, size)

scatter_plot_(range( 0, len(collation.crash_month) ), collation.crash_month, size)

scatter_plot_(range( 0, len(collation.crash_hour) ), collation.crash_hour, size)


def add_feature(feature, col_name):
    global collation
    collation[col_name] = pd.Series(data = feature)

feature = collation.crash_hour * collation.crash_month * collation.crash_week_day * 1000
add_feature(feature, 'mult_hourMonthWday')

feature = collation.precipitations + collation.snow
add_feature(feature, 'sum_precSnow')

feature =collation.jam_point * np.fabs(collation.temp)
add_feature(feature, 'mult_jamTemp')

feature = collation.jam_point * collation.snow
add_feature(feature, 'mult_jamSnow')

feature = np.tanh(collation.crash_hour * collation.crash_week_day * 1000)
add_feature(feature, 'tanhMult_hourMonthWday')

monthes = [row.tm_mon for row in list(map(num_to_dt, list(collation.time_)))]
hours = [row.tm_hour for row in list(map(num_to_dt, list(collation.time_)))]
week_days = [row.tm_wday for row in list(map(num_to_dt, list(collation.time_)))]

def scaling_data(list_, round_):
    return pd.Series(list( map(lambda x: round(x/100, round_), list_) ))

collation['crash_month'] = scaling_data(monthes, round_ = 2)
collation['crash_week_day'] = scaling_data(week_days, round_ = 2)
collation['crash_hour'] = scaling_data(hours, round_ = 2)
collation['lat'] = scaling_data(list(collation.lat), round_ = 6)
collation['lon'] = scaling_data(list(collation.lon), round_ = 6)

def merge_date_time(df):
    df['time_2'] = df.crash_date.astype(str).str.cat(df.crash_time.astype(str), sep=' ')
    drop_crash_date_column(df)

# ## crash_data
def add_crash_data(collation, crash_df):
    for item in crash_df.itertuples():
        row = {'time_': int(item.crash_time), 'lat': item.lat, 'lon': item.lon, 'crash_effect': item.crash_effect,
               'holyday_weight': None, 'jam_point': None, 'temp': None, 'soil_temp': None,  
               'pressure': None, 'pressure_change': None, 'humidity': None, 'wind_speed': None, 'cloudy': None, 
               'clouds_height': None, 'visibility': None, 'precipitation': None, 'snow': None, 'soil': None, 'weath_descr': None}
        collation.loc[item.Index] = row
    return collation
add_crash_data(collation, crash_df)

# ## holydays_data
def add_holydays_data(collation, hol_df):
    def get_hol_weight(time_):
        nonlocal hol_df
        for date, weight in zip(hol_df.date, hol_df.weight):
            if equal_dates(num_to_dt(date), num_to_dt(time_)):
                return weight
        return 0
    
    for indx, time_ in enumerate(collation.time_):       
        collation['holyday_weight'].loc[indx] = get_hol_weight(time_)
    
    return collation
add_holydays_data(collation, hol_df)

# ## jams_data
def add_jam_data(collation, jams_df):
    def get_jam_point(crash_time):
        nonlocal jams_df
        for row in jams_df.itertuples():
            if equal_week_times(row, num_to_dt(crash_time)):
                return row.point
        return 0
    
    for indx, time_ in enumerate(collation.time_):       
        collation['jam_point'].loc[indx] = get_jam_point(time_)
    
    return collation
add_jam_data(collation, jams_df)

# ## weather_data
def times_delta(time_1, time_2): 
    return fabs(time_1.tm_hour - time_2.tm_hour) * 60 + fabs(time_1.tm_min - time_2.tm_min)
    
def get_equal_day_times_dict(crash_time):
    global weath_df
    similar_times_dict = {}
    time_1 = num_to_dt(crash_time)
    for indx, time_ in enumerate(weath_df.time_):      
        time_2 = num_to_dt(time_)
        if equal_dates(time_1, time_2):
            similar_times_dict[indx] = times_delta(time_1, time_2)
            
    return similar_times_dict

def get_empty_params(row):
    logger = set()
    def check_param(param, name, zero_check=False):
        nonlocal logger
        if np.isnan(param) or zero_check and param == 0.:
            logger.add(name)
            
    check_param(row.pressure, 'pressure')
    check_param(row.pressure_change, 'pressure_change')
    check_param(row.visibility, 'visibility')
    check_param(row.soil, 'soil', zero_check=True)
    check_param(row.weath_descr, 'weath_descr', zero_check=True)
       
    return logger if logger else []

def sorted_dict_by_value(dict_):
    try:
        return sorted(dict_.items(), key = lambda x: x[1])
    except:
        return None
    
def get_built_weath_row(crash_time):
    sorted_times_dict = sorted_dict_by_value( get_equal_day_times_dict(crash_time) )
    curr_indx = sorted_times_dict[0][0]
    empty_params = get_empty_params(weath_df.loc[curr_indx])
    new_row = {}   
    for param in empty_params:
        for indx, time_del in sorted_times_dict[1:]:
            next_empty_params = get_empty_params(weath_df.loc[indx])
            if next_empty_params and param not in next_empty_params:
                new_row[param] = weath_df[param][indx]
                break
        if param not in new_row.keys():
            new_row[param] = weath_df[param][curr_indx]                
    for param in weath_df[curr_indx: curr_indx + 1]:
        if param not in empty_params:
            new_row[param] = weath_df[param][curr_indx]
            
    return new_row

weath_series = pd.Series(
    list(
        map(
            get_built_weath_row,
            crash_df.crash_time
        )
    )
)

for row in collation.itertuples():
    weath_row = get_built_weath_row(row.time_)
    collation['temp'].loc[row.Index] = weath_row['temp']
    collation['soil_temp'].loc[row.Index] = weath_row['soil_temp']
    collation['pressure'].loc[row.Index] = weath_row['pressure']
    collation['pressure_change'].loc[row.Index] = weath_row['pressure_change']
    collation['humidity'].loc[row.Index] = weath_row['humidity']
    collation['wind_speed'].loc[row.Index] = weath_row['wind_speed']
    collation['cloudy'].loc[row.Index] = weath_row['cloudy']
    collation['clouds_height'].loc[row.Index] = weath_row['clouds_height']
    collation['visibility'].loc[row.Index] = weath_row['visibility']
    collation['precipitation'].loc[row.Index] = weath_row['precipitations']
    collation['snow'].loc[row.Index] = weath_row['snow']
    collation['soil'].loc[row.Index] = weath_row['soil']
    collation['weath_descr'].loc[row.Index] = weath_row['weath_descr']

# ## other 
collation.time_ = collation.time_.astype('int')
collation.lat = np.round(collation.lat, decimals=6)
collation.lon = np.round(collation.lon, decimals=6)
collation.crash_effect = np.round( collation.crash_effect, decimals=4 )
collation.holyday_weight = np.round( collation.holyday_weight, decimals=4 )
collation.jam_point = np.round( collation.jam_point, decimals=4 )
collation.temp = np.round( collation.temp, decimals=4 )
collation.soil_temp = np.round( collation.soil_temp, decimals=4 )
collation.pressure = np.round( collation.pressure, decimals=4 )
collation.pressure_change = np.round( collation.pressure_change, decimals=4 )
collation.humidity = np.round( collation.humidity, decimals=4 )
collation.wind_speed = np.round( collation.wind_speed, decimals=4 )
collation.cloudy = np.round( collation.cloudy, decimals=4 )
collation.clouds_height = np.round( collation.clouds_height, decimals=4 )
collation.visibility = np.round( collation.visibility, decimals=4 )
collation.precipitation = np.round( collation.precipitation, decimals=4 )
collation.snow = np.round( collation.snow, decimals=4 )

def equal_dates(time_1, time_2):
    return time_1.tm_year == time_2.tm_year and            time_1.tm_mon == time_2.tm_mon and            time_1.tm_yday == time_2.tm_yday
        
def equal_week_times(jam_df_row, time_):
    week_convert = dict( zip(['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'], range(0,7)) )
    y = jam_df_row.year
    m = jam_df_row.month
    w = jam_df_row.week_day
    h = jam_df_row.hour
    return int(y == time_.tm_year) and \         
           int(m == time_.tm_mon) and \      
           int(week_convert[w] == time_.tm_wday) and \         
           int(h == time_.tm_hour)