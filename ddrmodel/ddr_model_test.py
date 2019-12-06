#Needs to output in true .sm format, but otherwise good

from shared_ddr_processing import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import random
import sys
import io
model = load_model("ddr_model.h5")

if len(sys.argv) > 1:
  path = sys.argv[1]
else:
  path = "../preprocessing/ddr_data/In The Groove/Anubis/c8_520.mnd"

np.set_printoptions(precision=4,suppress=True, floatmode="fixed")


raw_data = []
with open(path, mode='r', encoding="utf-8") as generic:
    songpath = generic.readline()
    bpm_data = generic.readline()
    offset = float(generic.readline())
    for line in generic:
        line_data = line.split()
        raw_data.append(tuple(int(x) for x in line_data))
jumps = 0
lrjumps = 0
udjumps = 0

bpm_list = process_bpm(bpm_data)
timepoints = []
blank_input = np.zeros(25) #9 mnd data, 16 step history
history = [blank_input]*LSTM_HISTORY_LENGTH
bpm_id = 0
bpm = bpm_list[bpm_id][1]
next_bpm_time = bpm_list[bpm_id+1][0]*48

now_sec = -offset
next_sec = now_sec
next_beat = raw_data[0][0]
now_beat = 0
minilast_beat = 0
while next_beat > next_bpm_time:
    assert(minilast_beat <= next_bpm_time)
    next_sec += (next_bpm_time-minilast_beat)/(bpm/60*48)
    minilast_beat = next_bpm_time
    bpm_id += 1
    bpm = bpm_list[bpm_id][1]
    next_bpm_time = bpm_list[bpm_id+1][0]*48
next_sec = (next_beat-minilast_beat)/(bpm/60*48)


notehistory = []
for i in range(len(raw_data)):
    (now_beat, note, start_long, end_long) = raw_data[i]
    last_beat = now_beat
    minilast_beat = now_beat
    last_sec = now_sec
    now_sec = next_sec
    next_beat = raw_data[i+1][0] if i+1 < len(raw_data) else now_beat+(192*5)
    while next_beat > next_bpm_time:
        assert(minilast_beat <= next_bpm_time)
        next_sec += (next_bpm_time-minilast_beat)/(bpm/60*48)
        minilast_beat = next_bpm_time
        bpm_id += 1
        bpm = bpm_list[bpm_id][1]
        next_bpm_time = bpm_list[bpm_id+1][0]*48
    next_sec += (next_beat-minilast_beat)/(bpm/60*48)
    beat_fract = beat_find(now_beat)
    mnd_in = [min(now_sec-last_sec,2)/2, min(next_sec-now_sec,2)/2,
                min((now_beat-last_beat)/384,1), min((next_beat-now_beat)/384,1),
                beat_fract/48, note/3, start_long/3, end_long/3, min(bpm/400,1)]
    parsed_mnd_in = np.array(mnd_in,dtype="float32",ndmin=2)
    hs = len(history)
    hist_in = np.array(history[hs-LSTM_HISTORY_LENGTH:hs],dtype="float32",ndmin=3)
    
    probs = model.predict([parsed_mnd_in,hist_in])
    
    (l,u,d,r) = (x[0].astype('float') for x in probs)
    available = {'l':l,'u':u,'d':d,'r':r}
    out = {'l':0,'u':0,'d':0,'r':0}
    for i, r in ((3, end_long),(2, start_long),(1, note)):
      for _ in range(r):
        best_v = 0
        for k, v in available.items():
          sample_v = v[i]*random.uniform(0.5,1)
          if sample_v > best_v:
            best_k = k
            best_v = sample_v
        out[best_k] = i
        del available[best_k]
    (L,U,D,R) = (out['l'],out['u'],out['d'],out['r'])
    #print([L,U,D,R],mnd_in)
    print([L,U,D,R],list(x[0].astype('float') for x in probs))
    tmp = list(map(lambda x: np.eye(4)[x],[L,U,D,R]))
    mnd_in.extend(np.ravel(tmp))
    history.append(mnd_in)
    notehistory.append([L,U,D,R])
    timepoints.append(now_beat)
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
            (l,u,d,r) = notehistory[i]
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
                #print(tmp,beat_offset*buf_res+curr_t)
            out.write(";\n" if (i == len(timepoints)) else ",\n")
            #clear current
            buf_res = 48
            buf_notes.clear()
            curr_t += 192
        buf_notes[t] = f"{l}{u}{d}{r}"
        res = beat_find(t)
        buf_res = min(res, buf_res)
        #print(l,u,d,r,t,beat_find(t),res)
print("Done!")