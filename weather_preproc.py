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

from math import pi, e

# ## Classes
class path:
    def __init__(self):
        self.RESOURCES = 'weather/resources/'
        self.STOP_WORDS = self.RESOURCES + 'stop_words'
        self.WEATHER_PREPROC = 'weather/'       
path = path()
    
class constants:
    def __init__(self):
        self.PI = pi
        self.E = e
        self.HASH_SPACE_FACTOR = pi ** pi // e * e
        self.SELECT_CENTRE = {
            'temp' : None,
            'soil_temp' : None,
            'pre' : None,
            'pre_ch' : None,
            'hum' : None,
            'wind' : None,
            'cld' : None,
            'cld_h' : None,
            'vis' : None,
            'prec' : None,
            'snow' : None
        }
        self.NONLIN_KOEFF = {
            'temp' : 0.0800,
            'soil_temp' : 0.0800,
            'pre' : 0.0680,
            'pre_ch' : 0.5000,
            'hum' : 0.0850,
            'wind' : 0.6670,
            'cld' : 0.0600,
            'cld_h' : 0.0027,
            'vis' : 0.9650,
            'prec' : 0.2800,
            'snow' : 0.0750,
            'default' : 0.0001
        }
constants = constants()

def df_to_csv(path = path.WEATHER_PREPROC, name = 'preproc_.csv'):
	df.to_csv(path + name, encoding='UTF-8')

def df_from_csv(path = path.WEATHER_PREPROC, name = 'preproc_.csv'):
	return pd.read_csv(path + name, encoding='UTF-8')

# ## Methods 
def dt_to_num(time_stamp):
    return int(time.mktime(time.strptime(time_stamp, '%d.%m.%Y %H:%M')))

def num_to_dt(date):
    return time.strftime( '%d.%m.%Y', time.gmtime(date + 3600*5) )

def get_selection_centre(selection, tag):
    """
    tags: 'temp', 'pre', 'pre_ch', 'hum', 'wind', 
    'cld', 'cld_h', 'vis', 'prec', 'snow'
    """
    def deprecated_selection_centre():
        nonlocal selection, tag
        return not ( (selection.min() + selection.max()) / 2 == constants.SELECT_CENTRE[tag] )

    if tag in constants.SELECT_CENTRE:
        if not constants.SELECT_CENTRE[tag] or deprecated_selection_centre():
            constants.SELECT_CENTRE[tag] = (selection.min() + selection.max()) / 2 
        return constants.SELECT_CENTRE[tag]
    else:
        return 0
    
def linear_normalization(selection = np.array([])):
    if selection.any():
        selection_max = selection.max()
        return pd.Series(list( map(lambda x: round(x, 4), selection/selection_max) )) 

def unsigned_nonlinear_normalization(selection = np.array([]), a = 0.1, tag = ''):
    centre = get_selection_centre(selection, tag)
    X_norm = 1 / (np.exp(-a * (selection - centre)) + 1)
    return X_norm

def value_unsigned_nonlinear_normalization(val = 0, selection = np.array([]), a = 0.1, tag = ''):
    centre = get_selection_centre(selection, tag)
    val_norm = 1 / (np.exp(-a * (val - centre)) + 1)
    return val_norm

def signed_nonlinear_normalization(selection = np.array([]), a = 0.1, tag = ''):
    centre = get_selection_centre(selection, tag)
    X_norm = (np.exp(a * (selection - centre)) - 1) / (np.exp(a * (selection - centre)) + 1)
    return X_norm

def value_signed_nonlinear_normalization(val = 0, selection = np.array([]), a = 0.1, tag = ''):
    centre = get_selection_centre(selection, tag)
    val_norm = (np.exp(a * (val - centre)) - 1) / (np.exp(a * (val - centre)) + 1)
    return val_norm

def unsigned_nonlinear_denormalization(X_norm = np.array([]), a = 0.1, tag = ''):
    if not isinstance(X_norm, np.ndarray):
        X_norm = np.array(list(X_norm))    
    if constants.SELECT_CENTRE[tag]:
        centre = constants.SELECT_CENTRE[tag]
        X_denorm = np.round(centre - (1/a * np.log(1/X_norm - 1)), decimals=6)
    else:
        X_denorm = []
    return X_denorm

def signed_nonlinear_denormalization(X_norm = np.array([]), a = 0.1, tag = ''):
    if not isinstance(X_norm, np.ndarray):
        X_norm = np.array(list(X_norm))
    if constants.SELECT_CENTRE[tag]: 
        centre = constants.SELECT_CENTRE[tag] 
        X_denorm = np.round(centre - (1/a * np.log( (1 - X_norm) / (1 + X_norm) )), decimals=6)
    else:
        X_denorm = []
    return X_denorm 

def nan_catcher(val, require_type):          
    if np.isnan(val):
        new = require_type(0) 
    else:
        new = require_type(val)
    
    return new

def numerization_time_values(series):
    return pd.Series(
        list (
            map (
                lambda x: dt_to_num(str(x)),
                list(series)
            ) 
        )
    )
            
def numerization_cloudy_values(series):
    num_values = []   
    for indx, val in  series.items():
        if isinstance(val, (int, float)):            
            num_values.append(nan_catcher(val, int)) 
            continue               
        tokens = [j.split('%') for j in val.split()]
        
        for tok in tokens[0]:
            if tok.isdigit():
                num_values.append(int(tok)) 
                break
            elif tok.split('–')[0].isdigit() and tok.split('–')[1].isdigit():
                left = int(tok.split('–')[0])
                right = int(tok.split('–')[1])
                num_values.append( int(np.mean([left, right])) )
                break
            elif tok.isalpha():
                num_values.append(0)
                break
    return pd.Series(num_values)
                 
def numerization_clouds_height_values(series):
    num_values = []   
    for indx, val in  series.items():
        if isinstance(val, (int, float)):            
            num_values.append(nan_catcher(val, int)) 
            continue          
        tokens = val.split('-') 
        
        if val in tokens:
            tok = val.split()[0]
            if tok.isdigit():
                num_values.append(int(tok))
                continue        
        elif tokens[0].isdigit() and tokens[1].isdigit():
            left = int(tokens[0])
            right = int(tokens[1])
            num_values.append( int(np.mean([left, right])) )       
        else:
            num_values.append(0)
            
    return pd.Series(num_values)

def numerization_precipitations_values(series):
    num_values = []
    for indx, val in  series.items():
        if isinstance(val, (int, float)):            
            num_values.append(nan_catcher(val, float)) 
            continue            
        elif val.split('.')[0].isdigit():
            num_values.append(float(val))
            continue       
        elif val.split()[0].isalpha():
            num_values.append(0.)
            continue
            
    return pd.Series(num_values)
        
def numerization_snow_values(series):
    num_values = []
    for indx, val in  series.items():
        if isinstance(val, (int, float)):            
            num_values.append(nan_catcher(val, float)) 
            continue            
        elif val.split('.')[0].isdigit():
            num_values.append(float(val))
            continue       
        elif val.split()[0].isalpha() and indx > 0:
            last_indx = indx - 1
            while last_indx >= 0:
                try:
                    curr_val = float(series[last_indx])
                except:
                    curr_val = ''
                if isinstance(curr_val, (int, float)) and curr_val>0:
                    num_values.append(curr_val)
                    break
                last_indx -= 1
        else:
            num_values.append(0.)
            
    return pd.Series(num_values)

def numerization_WD_values(series):
    corpus = get_uniq_filtred_docs(series)
    docs_hash_dict = get_docs_hashing(corpus)
    num_values = []
    
    for indx, doc in series.items():
        if isinstance(val, (int, float)):            
            num_values.append(nan_catcher(val, int)) 
            continue            
        if doc != ' ':         
            num_values.append( get_similar_doc_hash(doc, docs_hash_dict) )
            continue
        else:
            num_values.append(0)
            
    return pd.Series(num_values)

## WEATH CATEGORIAL
def filter_phrase(phrase, morph, stop_words): 
    if isinstance(phrase, (int, float)):
        return set()   
    tmp = phrase.replace('.', '').replace(',', '').lower() 
    tmp = re.sub(r"\d+", "", tmp, flags=re.UNICODE)
    tmp = re.sub(r'\([^)]+\)', "", tmp, flags=re.UNICODE)    
    tmp = ' '.join([morph.parse(word)[0].normal_form for word in tmp.split()]) 
    set_ = set()
    tmp = tmp.split()
    for word in tmp:
        if word not in stop_words:
            set_.add(word)           
    return set_
    
def filter_phrases(phrases):
    filtred = list()
    morph = pm2.MorphAnalyzer()
    stop_words = read_stop_words(path = path.STOP_WORDS) 
    
    for i, ph in enumerate(phrases):        
        if isinstance(ph, float):
            ph = phrases[i] = ''
            continue    
        filtred.append(filter_phrase(ph, morph, stop_words))        
    del phrases
    
    return filtred

def uniq_list(list_):
    tmp = [sorted(_) for _ in list_]
    unique_list = list(map(itemgetter(0), groupby(tmp)))   
    list_.clear()
    tmp.clear()   
    
    return unique_list

def list_to_string(list_):
    return ' '.join(list_)

def lists_to_strings(lists):
    return [list_to_string(list_) for list_ in lists]

def clear_doc_list(strings_list_):
    try:
        strings_ = list(set(strings_list_))
        strings_.remove('')
    except:
        pass
    finally:
        return strings_

def get_uniq_filtred_docs(docs_series_):
    docs = np.array(docs_series_)
    uniq_filtred = uniq_list(filter_phrases(docs))
    strings_ = lists_to_strings(uniq_filtred)
    strings_ = clear_doc_list(strings_)    
    return strings_

def get_filtred_doc(doc):
    morph = pm2.MorphAnalyzer()
    stop_words = read_stop_words(path = path.STOP_WORDS)    
    set_ = filter_phrase(doc, morph, stop_words)
    
    return list_to_string(list(set_))

def read_stop_words(path):
    stop_words = []
    try:
        with open(file=path, mode='rt', encoding='UTF-8') as fd:
            if fd.readable():
                sw = fd.read()
                for word in sw.split(','):
                    stop_words.append( word.strip().lower() )
    except FileNotFoundError:
        pass
    
    return stop_words

def tanimoto(doc_1, doc_2):
	"""
	distance metrics between documents
	"""
    a, b, c = len(doc_1.split()), len(doc_2.split()), 0.0
    small_doc, big_doc = (doc_1, doc_2) if a <= b else (doc_2, doc_1)    
    for word in small_doc.strip().split():
        if word in big_doc.strip().split():
            c += 1
            
    return c / (a + b - c)

def get_similar_doc(comp_doc, corpus):
    similarity = 0
    result = list(corpus)[0]   
    filtred = get_filtred_doc(comp_doc)
    for doc in corpus:
        if tanimoto(doc, filtred) > similarity:
            similarity = tanimoto(doc, filtred)          
            result = doc
            
    return result

def get_similar_doc_hash(comp_doc, docs_hash_dict):
    return docs_hash_dict[get_similar_doc(comp_doc, docs_hash_dict.keys())]

def get_docs_hashing(corpus):
    hash_space = int( len(corpus) * constants.HASH_SPACE_FACTOR )
    return { token : hash_ for token, hash_ in zip(corpus, 
                                                   np.array(list( map(get_doc_hashing , corpus) )) % hash_space
                                                  ) 
           }

sys.path.insert(0, path.RESOURCES)
from multimethod import multimethod as overload
@overload(str)
def get_doc_hashing(doc):
    return sum([hash(item) for item in doc.split()])

@overload(str, int)
def get_doc_hashing(doc, doc_context_length):
    hash_space = int(doc_context_length * constants.HASH_SPACE_FACTOR)
    return get_doc_hashing(doc) % hash_space

# ## Numerization
df.time_.update( numerization_time_values(df.time_) )
df.cloudy.update( numerization_cloudy_values(df.cloudy) )
df.clouds_height.update( numerization_clouds_height_values(df.clouds_height) )
df.precipitations.update( numerization_precipitations_values(df.precipitations) )
df.snow.update( numerization_snow_values(df.snow) )
df.weath_descr.update( numerization_WD_values(df.weath_descr) )
df.soil.update( numerization_WD_values(df.soil) )

# ## Normalization 
def plot_normalization_interval(selection_series, tag = 'default', step=0.1):
    selection_interval = np.arange(selection_series.min(), selection_series.max(), step)
    
    if tag == 'default':
        normalization_func = linear_normalization
    elif tag in ['temp', 'soil_temp', 'pre_ch']:
        normalization_func = signed_nonlinear_normalization
    else:
        normalization_func = unsigned_nonlinear_normalization
    
    if 'nonlinear' in str(normalization_func):
        normalization_interval = normalization_func(selection_interval, constants.NONLIN_KOEFF[tag], tag=tag)
    else:
        normalization_interval = normalization_func(selection_interval)
        
    plt.plot(
        selection_interval, 
        normalization_interval, 
        'g-'
    )
    plt.rcParams['figure.figsize'] = (5,3)
    plt.show()
    
def plot_normalization_intervals(df, tags = constants.NONLIN_KOEFF.keys()):
    colums = ['temp', 'soil_temp', 'pressure', 'pressure_change', 'humidity', 'wind_speed', 'cloudy',               'clouds_height', 'visibility', 'precipitations', 'snow', 'weath_descr']
    
    for colum, tag in zip(colums, tags):
        print(tag, constants.NONLIN_KOEFF[tag])
        plot_normalization_interval(df[colum], tag)
plot_normalization_intervals(df)

df.temp.update( signed_nonlinear_normalization(selection = df.temp, a = constants.NONLIN_KOEFF['temp'], tag = 'temp') )
df.soil_temp.update( signed_nonlinear_normalization(selection = df.soil_temp, a = constants.NONLIN_KOEFF['soil_temp'], tag = 'soil_temp') )
df.pressure.update( unsigned_nonlinear_normalization(selection = df.pressure, a = constants.NONLIN_KOEFF['pre'], tag = 'pre') )
df.pressure_change.update( signed_nonlinear_normalization(selection = df.pressure_change, a = constants.NONLIN_KOEFF['pre_ch'], tag = 'pre_ch') )
df.humidity.update( unsigned_nonlinear_normalization(selection = df.humidity, a = constants.NONLIN_KOEFF['hum'], tag = 'hum') )
df.wind_speed.update( unsigned_nonlinear_normalization(selection = df.wind_speed, a = constants.NONLIN_KOEFF['wind'], tag = 'wind') )
df.cloudy.update( unsigned_nonlinear_normalization(selection = df.cloudy, a = constants.NONLIN_KOEFF['cld'], tag = 'cld') )
df.clouds_height.update( unsigned_nonlinear_normalization(selection = df.clouds_height, a = constants.NONLIN_KOEFF['cld_h'], tag = 'cld_h') )
df.visibility.update( unsigned_nonlinear_normalization(selection = df.visibility, a = constants.NONLIN_KOEFF['vis'], tag = 'vis') )
df.precipitations.update( unsigned_nonlinear_normalization(selection = df.precipitations, a = constants.NONLIN_KOEFF['prec'], tag = 'prec') )
df.snow.update( unsigned_nonlinear_normalization(selection = df.snow, a = constants.NONLIN_KOEFF['snow'], tag = 'snow') )
df.soil.update( linear_normalization(selection = df.soil) )
df.weath_descr.update( linear_normalization(selection = df.weath_descr) )
