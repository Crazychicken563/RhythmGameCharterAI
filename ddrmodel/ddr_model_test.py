#NOT YET COMPLETE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import numpy as np
import random
import sys
import io
model = load_model("ddr_model.h5")

LSTM_HISTORY_LENGTH = 32

if len(sys.argv) > 1:
  path = sys.argv[1]
else:
  path = "../preprocessing/ddr_data/In The Groove/Anubis/c6_5820.mnd"

np.set_printoptions(precision=4,suppress=True)
with open(path, mode='r', encoding="utf-8") as generic:
    songpath = generic.readline()
    bpm = generic.readline()
    offset = generic.readline()
    last_time = 0
    history = [[0, 0, 0, 0]]*LSTM_HISTORY_LENGTH
    while True:
        line = generic.readline()
        if len(line) == 0:
            break
        line_data = line.split()
        (time_point, note, start_long, end_long) = (int(x) for x in line_data)
        for t in [48, 24, 16, 12, 8, 6, 4, 3, 2, 1]: #time_resolution of individual notes
            if time_point%t == 0:
                beat_fract = t
                break
        mnd_in = np.array([time_point-last_time, beat_fract, note, start_long, end_long],dtype="float32",ndmin=2)
        last_time = time_point
        hs = len(history)
        hist_in = np.array(history[hs-LSTM_HISTORY_LENGTH:hs],dtype="float32",ndmin=3)
        probs = model.predict([mnd_in,hist_in])
        (l,u,d,r) = (x[0].astype('float') for x in probs)
        available = {'l':l,'u':u,'d':d,'r':r}
        out = {'l':0,'u':0,'d':0,'r':0}
        for _ in range(end_long):
            best_v = 0
            for k, v in available.items():
                if v[3] > best_v:
                    best_k = k
                    best_v = v[3]
            out[best_k] = 3
            del available[best_k]
        for _ in range(start_long):
            best_v = 0
            for k, v in available.items():
                if v[2] > best_v:
                    best_k = k
                    best_v = v[2]
            out[best_k] = 2
            del available[best_k]
        for _ in range(note):
            best_v = 0
            for k, v in available.items():
                if v[1] > best_v:
                    best_k = k
                    best_v = v[1]
            out[best_k] = 1
            del available[best_k]
        (L,U,D,R) = (out['l'],out['u'],out['d'],out['r'])
        print(mnd_in,[L,U,D,R])
        print(list(x[0].astype('float') for x in probs))
        history.append([L,U,D,R])