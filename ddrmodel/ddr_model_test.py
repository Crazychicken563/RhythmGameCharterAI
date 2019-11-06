#Needs to output in true .sm format, but otherwise good

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

LSTM_HISTORY_LENGTH = 64

if len(sys.argv) > 1:
  path = sys.argv[1]
else:
  path = "../preprocessing/ddr_data/In The Groove/Anubis/c6_5820.mnd"

np.set_printoptions(precision=4,suppress=True)

def beat_find(time_point):
    for tt in [48, 24, 16, 12, 8, 6, 4, 3, 2, 1]: #time_resolution of individual notes
        if time_point%tt == 0:
            return tt
    raise Exception(f"beat_frac {time_point} not int")

jumps = 0
lrjumps = 0
udjumps = 0
timepoints = []
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
        beat_fract = beat_find(time_point)
        mnd_in = np.array([time_point-last_time, beat_fract, note, start_long, end_long, bpm],dtype="float32",ndmin=2)
        last_time = time_point
        hs = len(history)
        hist_in = np.array(history[hs-LSTM_HISTORY_LENGTH:hs],dtype="float32",ndmin=3)
        probs = model.predict([mnd_in,hist_in])
        (l,u,d,r) = (x[0].astype('float') for x in probs)
        available = {'l':l,'u':u,'d':d,'r':r}
        out = {'l':0,'u':0,'d':0,'r':0}
        for i, r in ((3, end_long),(2, start_long),(1, note)):
          for _ in range(r):
              best_v = 0
              for k, v in available.items():
                  if v[i] > best_v:
                      best_k = k
                      best_v = v[i]
              out[best_k] = i
              del available[best_k]
        (L,U,D,R) = (out['l'],out['u'],out['d'],out['r'])
        print(mnd_in,[L,U,D,R])
        #print(list(x[0].astype('float') for x in probs))
        history.append([L,U,D,R])
        timepoints.append(time_point)
        if (note+start_long == 2):
            jumps += 1
            if (U>0 and D>0):
                udjumps += 1
            if (L>0 and R>0):
                lrjumps += 1
print(jumps,udjumps,lrjumps)
with open("output.sm", mode='w', encoding="utf-8") as out:
    curr_t = 0
    buf_notes = {}
    buf_res = 48
    for i in range(len(timepoints)+1):
        if i == len(timepoints):
            (l,u,d,r) = (0,0,0,0)
            t = curr_t+192
        else:
            (l,u,d,r) = history[LSTM_HISTORY_LENGTH+i]
            t = timepoints[i]
        if (curr_t+192 <= t):
            #blank measures
            while(curr_t+(192*2)) <= t:
                out.write("0000\n0000\n0000\n0000\n,\n")
                curr_t += 192
            #print current info
            for beat_offset in range(192//buf_res):
                tmp = buf_notes.get(beat_offset*buf_res+curr_t, "0000")
                out.write(tmp+"\n")
                print(tmp,beat_offset*buf_res+curr_t)
            out.write(",\n")
            print(",")
            #clear current
            buf_res = 48
            buf_notes.clear()
            curr_t += 192
        buf_notes[t] = f"{l}{u}{d}{r}"
        res = beat_find(t)
        buf_res = min(res, buf_res)
        print(l,u,d,r,t,beat_find(t),res)
print("Done!")