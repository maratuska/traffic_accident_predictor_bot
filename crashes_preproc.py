# coding: utf-8
import pandas as pd
import numpy as np
import collections as clcts
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt
import geoplotlib
import random as rnd

crashes_path = {
    16 : 'crash/2016/2016.csv', 
    17 : 'crash/2017/2017.csv', 
    18 : 'crash/2018/2018.csv'
}

def df_to_csv(path = 'crash/', name = 'preproc_.csv'):
    df.to_csv(path + name, encoding='UTF-8')

def df_from_csv(path = 'crash/', name = 'preproc_.csv'):
    return pd.read_csv(path + name, encoding='UTF-8')

def scatter_plot_(X = None, Y = None, size = (27, 7)):
    plt.grid(True, alpha = 0.2) 
    plt.scatter(X, Y, color='g', alpha = 0.7)
    plt.plot(X, Y, alpha = 0.3)
    plt.rcParams['figure.figsize'] = (size[0], size[1])
    plt.show()

def exception_passive_catcher(operation):
    def wrapper(*args):
        try:            
            return operation(*args)
        except:
            pass
    return wrapper

def read_card(crash_path):
    data = pd.read_csv(crash_path, encoding='UTF-8') 
    return data

def read_cards(crashes_path):
    cards_collection = {}
    for indx, path in crashes_path.items():
        cards_collection[indx] = read_card(path)
    return cards_collection

def get_cards_by_dist(reg_code, distinct, cards):
    local = pd.DataFrame(columns=['crash_date', 'crash_time', 'lat', 'lon', 'addr', 'participants', 'fat', 'vic'])
    for indx, data in enumerate(zip(cards.reg_code, cards.address)):
        code = data[0] 
        addr = data[1]
        if int(code) == int(reg_code) and str(distinct).lower() in addr.lower():
            local.loc[len(local)] = cards.crash_date[indx], cards.crash_time[indx], cards.latitude[indx], \                               
            cards.longitude[indx], cards.address[indx], cards.participants_amount[indx], \                                   
            cards.fatalities_amount[indx], cards.victims_amount[indx]
    return local.copy()
    
def get_cards_collection_by_dist(reg_code, distinct, cards_collection):
    dist_cards_collection = {}    
    for year, cards in cards_collection.items():
        dist_cards_collection[year] = get_cards_by_dist(reg_code, distinct, cards)
    return dist_cards_collection

def concat_cards(cards_collection):
    return pd.concat(list(cards_collection.values()), ignore_index=True)

@exception_passive_catcher
def merge_date_time(df):
    df['crash_time'] = df.crash_date.astype(str).str.cat(df.crash_time.astype(str), sep=' ')
    drop_crash_date_column(df)

@exception_passive_catcher
def drop_crash_date_column(df):
    df.drop(labels=['crash_date'], axis=1, inplace=True)
    
@exception_passive_catcher
def drop_addr_column(df):
    df.drop(labels=['addr'], axis=1, inplace=True)

def dt_to_num(time_):
    time_stamp = time.strptime(time_, '%Y%m%d %H:%M')
    return int(time.mktime(time_stamp))

def numerization_time_values(series):
    return pd.Series(
        list (
            map (
                dt_to_num,
                list(series)
            ) 
        )
    )

def draw_crashes_points(cards, filtred = True):
    if filtred: 
        latitude, longitude = ('lat', 'lon')
    else: 
        latitude, longitude = ('latitude', 'longitude')
        
    points = {'lon':[], 'lat':[]}
    for lat, lon in zip(cards[latitude], cards[longitude]):
        points['lon'].append(float(lon))
        points['lat'].append(float(lat))

    geoplotlib.dot(points, point_size=2, color='green')
    geoplotlib.show()
    
# # Crash metric
# ## $ \frac{W0 (W1 (w0 (w1 + 1) * fat + w1 * vic) + W2 (w2 * nvic) )}{prtcpnt}$
# #### $W0 = 1 + 0.3333 fat$
# #### $W1 = 1 + 0.1000 prtcpnt$
# #### $W2 = 1 + 0.0100 prtcpnt$
# #### $w0 = (1.1700 : 1.1900)$
# #### $w1 = \frac{1}{2} + (-0.0100 : 0.0100)$
# #### $w2 = (0.0050 : 0.0090)$

def get_crash_effect(prtcpnt, fat, vic):
    if prtcpnt < fat + vic:
        prtcpnt = fat + vic
    
    n_vic = prtcpnt - (fat + vic)

    W0_wrp = 1 + (fat * 0.3333)
    W1_wrp = 1 + (prtcpnt * 0.1000)
    W2_wrp = 1 + (prtcpnt * 0.0100)
    w0 = rnd.uniform(1.1700, 1.1900)
    w1 = 1/2 + rnd.uniform(-0.0100, 0.0100)
    w2 = rnd.uniform(0.0050, 0.0090)
    
    result = W0_wrp * (W1_wrp * (w0 * (w1+1) * fat + w1 * vic) + W2_wrp * (w2 * n_vic)) / prtcpnt    
    return round(result, 4)

@exception_passive_catcher
def compute_crash_effects(df):
    if 'participants' and 'fat' and 'vic' in df.columns:
        return pd.Series(
            list (
                map (
                    lambda x, y, z: get_crash_effect(x, y, z),
                    df.participants, df.fat, df.vic
                ) 
            )
        )
    else:
        return None

def merge_crash_indicators(df):  
    result = compute_crash_effects(df)
    if isinstance(result, pd.Series):
        df['crash_effect'] = result
    drop_crash_indicators(df)

@exception_passive_catcher
def drop_crash_indicators(df):
    df.drop(['participants', 'fat', 'vic'], axis=1, inplace=True)
                
cards_collection = read_cards(crashes_path)
ufa_cards_collection = get_cards_collection_by_dist(80,'уфа', cards_collection)
df = concat_cards(ufa_cards_collection)

merge_date_time(df)
merge_crash_indicators(df)
drop_addr_column(df)

df.crash_time.update( numerization_time_values(df.crash_time) )
draw_crashes_points(df)

