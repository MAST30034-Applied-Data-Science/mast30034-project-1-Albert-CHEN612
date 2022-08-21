import pandas as pd
from pandas import read_parquet
import os
import matplotlib.pyplot as plt
import io
import numpy as np
import random
import pandas as pd
import os
import keras
import csv

import datetime

from keras import layers
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers import BatchNormalization
from keras.preprocessing import sequence
from sklearn.cluster import AffinityPropagation, KMeans
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Dropout, concatenate
from keras.models import Model, Sequential
# from matrix_form import tweets_read
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.preprocessing import normalize
from keras.layers import Input, Embedding, LSTM, Dense
from sklearn.manifold import LocallyLinearEmbedding
from collections import Counter
# from nltk.corpus import stopwords
import re

import sklearn

from numpy import linalg

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from collections import Counter

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import sklearn.preprocessing

# loading all taxi data from dir and calculate the total count of trips.

def dataloading(dirpath):
    filenamelist = os.listdir(dirpath)
    ncount = 0
    data = dict()
    for filename in filenamelist:
        print(filename)
        datatmp = read_parquet(dirpath + filename)
        ncount = ncount + datatmp.count()
        print(datatmp.head())
        data[filename] = datatmp
    return data, ncount


# calculating the trip frequency per hour in one day
def trips_time_frequency(parquet_data):
    times = parquet_data['tpep_pickup_datetime']
    hours = dict()
    for i in times:
        hour = int(str(i)[11:13])
        if hour in hours:
            hours[hour] = hours[hour] + 1
        else:
            hours[hour] = 1
    print(hours)
    print(hours.values())


def trips_nyc_loc_frequency(data, time):
    dataloc = data['PULocationID']
    times = data['tpep_pickup_datetime']
    locs = dict()
    for i in range(len(dataloc)):
        loc = int(dataloc[i])
        hour = int(str(times[i])[11:13])
        print(hour)
        print(time)
        if hour == time:
            if loc in locs:
                locs[loc] = locs[loc] + 1
            else:
                locs[loc] = 1
    print(locs)
    print(locs.values())



def get_lstm_input_output(part_name,vocab_size):
    main_input = Input(shape=(MAX_LEN,), dtype='int32', name=part_name+'_input')
    x = Embedding(output_dim=EMBED_SIZE, input_dim=vocab_size, input_length=MAX_LEN)(main_input)
    lstm_out = LSTM(HIDDEN_SIZE)(x)
    return main_input,lstm_out


def matrix_construction(data):
    dataloc = data['PULocationID']
    drloc=data['DOLocationID']
    times = data['tpep_pickup_datetime']
    matrix=[[0]*(265)]*(265)

    for i in range(len(dataloc)):
        puloc=dataloc[i]
        matrix[dataloc[i]][drloc[i]]=matrix[dataloc[i]][drloc[i]]+1
    return matrix



def time_segmentation(data):
    times = data['tpep_pickup_datetime']
    dataloc_pu = data['PULocationID']
    dataloc_dr = data['DOLocationID']
    mp=[]
    dt=[]
    ep=[]
    nt=[]
    for i in range(len(dataloc_pu)):
        loc_pu = int(dataloc_pu[i])
        loc_dr = int(dataloc_dr[i])
        hour = int(str(times[i])[11:13])
        if hour >=6 and hour <9:
            mp.append((loc_pu, loc_dr))
        elif hour >=9 and hour<17:
            dt.append((loc_pu, loc_dr))
        elif hour >= 17 and hour < 21:
            ep.append((loc_pu, loc_dr))
        elif hour > 17 or hour <6:
            nt.append((loc_pu, loc_dr))

    all=dict()
    all['mp']=mp
    all['dt'] = dt
    all['ep'] = ep
    all['nt'] = nt
    return all


def loc_frequency_seg(data_list):
    locs = dict()

    for i in range(len(data_list)):
        loc_pu = data_list[i][0]
        loc_dr = data_list[i][1]
        if loc_pu in locs:
            locs[loc_pu] = locs[loc_pu] + 1
        else:
            locs[loc_pu] = 1
    print(locs)
    print(locs.values())


data, ncount = dataloading('yellow/test/')


print(ncount)
for filename in data:
    all_data=time_segmentation(data[filename])
    print(all_data)
    res=dict()
    for i in all_data['mp']:
        print(i)
        if i in res:
            res[i]=res[i]+1
        else:
            res[i]=1
    print(res)


for filename in data:
    all_data=matrix_construction(data[filename])
    print(all_data)
    res=dict()
    for i in all_data['mp']:
        print(i)
        if i in res:
            res[i]=res[i]+1
        else:
            res[i]=1
    print(res)



HIDDEN_SIZE=16
MAX_LEN=265
EMBED_SIZE=32
timestep=64
maxlen=265

from sklearn.model_selection import train_test_split
y=list(np.arange(1,265))

X_train, X_test, y_train, y_test  = train_test_split(all_data, y, test_size=0.2, random_state=1)

model = keras.Sequential()
model.add(Embedding(maxlen, EMBED_SIZE, input_length=MAX_LEN, dropout=0.2))
model_input=keras.Input(shape=(timestep, maxlen))
model_lstm=layers.LSTM(HIDDEN_SIZE)(model_input)
model_output=layers.Dense(1, activation="sigmoid")(model_lstm)
model = keras.Model(input=model_input,output=model_output)
optimizer = keras.optimizers.RMSprop(lr=0.001)
model.compile(loss="mean_squared_error", optimizer=optimizer)

batch_size=256 # lstm
print(model.summary())
model.fit(X_train, y_train, batch_size=batch_size, epochs=500,verbose=2)
t_strat=datetime.datetime.now()
ynew=model.predict(X_test)

