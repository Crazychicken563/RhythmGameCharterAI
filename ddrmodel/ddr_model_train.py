import os
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback
#from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import random
import sys
import io
#Note: Requires both ddr_finder and ddr_to_generic to be run first
#will dynamically construct both the MND's data and the output to compare against
#Current input is purely the previous arrows' positions/type (basic arrow, freeze start, freeze end, none)x4
#none is used only for start of song data
#Later revision will use the actual song data as well, but that is not included in this prototype
source_dir = "../preprocessing/ddr_data/In The Groove"
song_count = 0
song_path = []
song_mnd = []
song_output = []
def setup_data():
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        name = os.path.relpath(dirpath, source_dir)
        if len(dirnames) > 5:
            print(">Subdirectories found in "+name+", continuing.")
            continue
        if dirnames:
            print("----<5 subdirectories in "+name+", investigate!")
        if not "chart_data.dat" in filenames:
            print("Chart data not found! "+name)
            continue
        with open(os.path.join(dirpath, "chart_data.dat"), encoding="utf-8") as chart_data:
            (a, chart_path) = chart_data.readline().strip().split("=")
            (b, music_path) = chart_data.readline().strip().split("=")
            (c, bpm)        = chart_data.readline().strip().split("=")
            (d, offset)     = chart_data.readline().strip().split("=")
            assert (a, b, c, d) == ("CHART", "MUSIC", "BPM", "OFFSET")
            with open(chart_path, encoding="latin-1") as chart:
                while True:
                    tmp = chart_data.readline()
                    if len(tmp) < 3:
                        break
                    (difficulty, position) = tmp.strip().split("@")
                    difficulty = difficulty.strip(":")
                    chart.seek(int(position))
                    this_difficulty_file = os.path.join(dirpath,"c"+str(difficulty)+"_"+position+".mnd")
                    if os.path.exists(this_difficulty_file):
                        song_path.append(this_difficulty_file)
                        assert(mnd_data(chart))


def mnd_data(chart):
    #Assumes chart is valid and was seek()'d to already
    time_point = 0
    stored_lines = []
    mnd_data = []
    output = []
    last_time = 0
    while True:
        line = chart.readline()
        if len(line) == 0:
            print("Unexpected EoF!")
            return False
        line = line.strip()
        if ";" in line or "," in line:
            #process a measure
            count = len(stored_lines)
            if not count in [4, 8, 12, 16, 24, 32, 48, 64, 96, 192]:
                print(f"bad count({count}) at {chart.tell()}")
                return False
            time_resolution = 192//count #integer division
            for notes in stored_lines:
                note = notes.count("1")
                #Does nothing with mines, maybe make them notes? ("M")
                start_long = notes.count("2")+notes.count("4")
                #Counting rolls (4) as long notes
                end_long = notes.count("3")
                if (note or start_long or end_long):
                    notes = notes.replace("M","0").replace("4","2")
                    tmp = list(map(lambda x: to_categorical(x, num_classes=4,dtype="int32"),notes))
                    output.append(tmp)
                    for t in [48, 24, 16, 12, 8, 6, 4, 3, 2, 1]: #time_resolution of individual notes
                        if time_point%t == 0:
                            beat_fract = t
                            break
                    else:
                        print("beat_fract nonsensical (nonint?)")
                        return False
                    mnd_data.append([time_point-last_time, beat_fract, note, start_long, end_long])
                    last_time = time_point
                time_point += time_resolution
            stored_lines = []
        else:
            if len(line) == 4:
                stored_lines.append(line)
        if ";" in line:
            song_mnd.append(mnd_data)
            song_output.append(output)
            return True

setup_data()

LSTM_HISTORY_LENGTH = 32

mnd_input = layers.Input(shape=(5,),name="mnd_input")
x = layers.Dense(32)(mnd_input)
hist_input = layers.Input(shape=(LSTM_HISTORY_LENGTH-1,4,),name="hist_input")
hist_lstm = layers.LSTM(32)(hist_input)
x = layers.concatenate([x,hist_lstm])
x = layers.Dense(32)(x)
x = layers.Dense(32)(x)
outL = layers.Dense(4, activation='softmax', name = "outL")(x)
outU = layers.Dense(4, activation='softmax', name = "outU")(x)
outD = layers.Dense(4, activation='softmax', name = "outD")(x)
outR = layers.Dense(4, activation='softmax', name = "outR")(x)

optimizer = RMSprop(learning_rate=0.01)
model = models.Model(inputs=[mnd_input,hist_input],outputs=[outL,outU,outD,outR])

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def generate_song_inout_data(mnd_arr, out_arr):
    assert(len(mnd_arr) == len(out_arr))
    inputs = []
    outputs = []
    maxouts = []
    fulldata = []
    for i in range(LSTM_HISTORY_LENGTH-1):
        inputs.append(np.array([0, 0, 0, 0, 0]))
        maxouts.append([0, 0, 0, 0])
        outputs.append(to_categorical([0, 0, 0, 0],4))
    for pos in range(len(mnd_arr)):
        i = pos+LSTM_HISTORY_LENGTH-1
        inputs.append(mnd_arr[pos])
        outputs.append(out_arr[pos])
        maxouts.append(np.argmax(out_arr[pos],axis=1))
        yield ((np.array(inputs[i]), np.array(maxouts[pos:i])), np.array(outputs[i]))

for i in range(100): #true epoch count AKA number of songs to process
    print(song_path[i])
    all_data = generate_song_inout_data(song_mnd[i], song_output[i])
    (ins, outs) = zip(*all_data)
    (in_mnd, in_hist) = zip(*ins)
    (outL, outU, outD, outR) = zip(*outs)
    model.fit(x=[in_mnd,in_hist], y=[outL, outU, outD, outR])

model.save("ddr_model.h5")