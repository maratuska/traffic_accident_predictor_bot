# coding: utf-8
import xgboost
from xgboost import XGBRegressor as xgbr

import keras
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import LeakyReLU
from keras import layers
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D

from matplotlib.pyplot import (axes,axis,title,legend,figure,
                               xlabel,ylabel,xticks,yticks,
                               xscale,yscale,text,grid,
                               plot,scatter,errorbar,hist,polar,
                               contour,contourf,colorbar,clabel,
                               imshow)
from mpl_toolkits.mplot3d import Axes3D
from numpy import (linspace,logspace,zeros,ones,outer,meshgrid,
                   pi,sin,cos,sqrt,exp)
from numpy.random import normal

from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import SVG
from collator import collation

def learning_plot(history):
  #keras
  get_ipython().magic('matplotlib inline')
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(history.epoch, history.history["loss"], label="loss")
  plt.plot(history.epoch, history.history["acc"], label="acc")
  plt.plot(history.epoch, history.history["mean_absolute_error"], label="mae")

def save_model(model, name)
  #keras
  model_json = model.to_json()
  json_file = open("NN/{}.json".format(name), "w")
  json_file.write(model_json)
  json_file.close()
  model.save_weights("NN/{}.h5".format(name))

def load_model(name)
  #keras
  json_file = open("NN/{}.json".format(name), "r")
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = keras.models.model_from_json(loaded_model_json)
  loaded_model.load_weights("NN/{}.h5".format(name))

def choose_xgb_config:
  #xgboost
  _tmp_config = {
    'acc' : 0,
    'dep' : None,
    'est' : None
  }
  for depth_ in [4,6,8,10,12,16,20,32,64]: 
      for est in [100,200,300,500,700,1000,1200,1400,1500]: 
          model = xgbr(max_depth=depth_,
                       n_estimators=est,
                       eval_metric='rmse', 
                       n_jobs=4,
                       learning_rate=0.18
          )
          model.fit(X_train, Y_train)
          accuracy = model.score(X_test, Y_test)
          if accuracy > _tmp_config['acc']:
            _tmp_config['acc'], _tmp_config['dep'], _tmp_config['est'] = (accuracy, depth_, est)
          print("Accuracy: {}, dep: {}, est: {}".format(accuracy * 100.0, depth_, est))
    return _tmp_config

X = collation[['lat', 'lon', 'holyday_weight', 'jam_point', 'temp', 'soil_temp', 'pressure', 'pressure_change', 'humidity', 'wind_speed', 'cloudy', 'clouds_height', \
                     'visibility', 'precipitations', 'snow', 'soil', 'weath_descr', 'crash_month', 'crash_week_day', 'crash_hour', 'mult_hourMonthWday', 'sum_precSnow', \
                     'mult_jamTemp', 'mult_jamSnow', 'tanhMult_hourMonthWday']]
Y = collation[['total_crash_point']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

model_5 = Sequential()
model_5.add(Dense(1024, activation='sigmoid', input_dim=25))
model_5.add(Dense(700)) 
model_5.add(LeakyReLU(alpha = 0.12)) 
model_5.add(Dense(512))
model_5.add(LeakyReLU(alpha = 0.1))
model_5.add(Dense(512))
model_5.add(LeakyReLU(alpha = 0.1))
model_5.add(Dense(256, activation='relu'))
model_5.add(Dense(64, activation='relu'))
model_5.add(Dense(1))
model_5.add(Activation('linear'))
model_5.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy', 'mse', 'mae'])

history_5 = model_5.fit(
    X_train, y_train, 
    batch_size = 16,
    epochs = 2000
)

img_rows, img_cols = (25, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
model_3 = Sequential()
model_3.add(Conv2D(200, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model_3.add(MaxPooling2D(pool_size=(1, 1)))
model_3.add(Dropout(0.2))
model_3.add(Conv2D(100, (2, 2), activation='relu'))
model_3.add(MaxPooling2D(pool_size=(1, 1)))
model_3.add(Dropout(0.2))
model_3.add(Conv2D(50, (2, 1), activation='relu'))
model_3.add(MaxPooling2D(pool_size=(1, 1)))
model_3.add(Dropout(0.2))
model_3.add(Flatten())
model_3.add(Dense(1024, activation='relu'))
model_3.add(Dense(512, activation='relu'))
model_3.add(Dropout(0.3))
model_3.add(Dense(1, activation='linear'))
model_3.compile(loss="mse", 
              optimizer="adam", 
              metrics=["accuracy", "mse"])
print(model_3.summary())

history_3 = model_3.fit(
    X_train, y_train, 
    batch_size = 16,
    epochs = 2000
)

model = xgbr(max_depth = 8,
             n_estimators = 118,
             eval_metric = 'rmse', 
             n_jobs = 4,
             learning_rate = 0.2
)
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)
print("Accuracy: %.4f%%" % (accuracy * 100.0))

plt.rcParams['figure.figsize'] = (30, 30)
xgboost.plot_tree(model, num_trees=1)
plt.show()
