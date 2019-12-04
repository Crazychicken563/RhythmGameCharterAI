import os
import re

import random
import sys
import io
import pickle
import math
from shared_ddr_processing import *
import multiprocessing
from multiprocessing import Pool
#Note: Requires both ddr_finder and ddr_to_generic to be run first
#will dynamically construct both the MND's data and the output to compare against
#Current input is the previous arrows' positions/type (basic arrow, freeze start, freeze end, none)x4, MND data for both past arrows and the new step, and a very small amount of audio data around the new step
#start of song data has a ton of 0 (except for BPM)
source_dir = "../preprocessing/ddr_data"
songs_per = 1024

#Returns a list of charts to be used later (this is cached in ddr_dataset.p)
def song_data(skipto = ""):
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        if skipto != "":
            if skipto in dirpath:
                skipto = ""
            else:
                continue
        name = os.path.relpath(dirpath, source_dir)
        print(name)
        if len(dirnames) > 5:
            print(">Subdirectories found in "+name+", continuing.")
            continue
        if dirnames:
            print("----<5 subdirectories in "+name+", investigate!")
        if not "chart_data.dat" in filenames:
            print("Chart data not found! "+name)
            continue
        with open(os.path.join(dirpath, "chart_data.dat"), encoding="utf-8") as chart_data:
            (a, chart_path) = chart_data.readline().strip().split(":",1)
            (b, music_path) = chart_data.readline().strip().split(":",1)
            (c, bpm)        = chart_data.readline().strip().strip(";").split(":")
            (d, offset)     = chart_data.readline().strip().split(":")
            assert (a, b, c, d) == ("CHART", "MUSIC", "BPM", "OFFSET")
            with open(chart_path, encoding="latin-1") as chart:
                while True:
                    tmp = chart_data.readline()
                    if len(tmp) < 3:
                        break
                    (difficulty, position) = tmp.strip().split("@")
                    difficulty = difficulty.strip(":")
                    this_difficulty_file = os.path.join(dirpath,"c"+str(difficulty)+"_"+position+".mnd")
                    if os.path.exists(this_difficulty_file):
                        yield (bpm,float(offset),int(position),dirpath,chart_path,music_path)


def mnd_getdata(chart, flipLR=False, flipUD=False):
    #Assumes chart is valid and was seek()'d to already
    time_point = 0
    stored_lines = []
    mnd_data = []
    output = []
    last_time = 0
    while True:
        line = chart.readline()
        if len(line) == 0:
            raise Exception("Unexpected EoF!")
        line = line.strip()
        if ";" in line or "," in line:
            #process a measure
            count = len(stored_lines)
            if not count in [4, 8, 12, 16, 24, 32, 48, 64, 96, 192]:
                raise Exception(f"bad count({count}) at {chart.tell()}")
            time_resolution = 192//count #integer division
            for notes in stored_lines:
                note = notes.count("1")
                #Does nothing with mines, maybe make them notes? ("M")
                start_long = notes.count("2")+notes.count("4")
                #Counting rolls (4) as long notes
                end_long = notes.count("3")
                if (note or start_long or end_long):
                    notes = notes.replace("M","0").replace("4","2").replace("K","0").replace("L","0").replace("F","0")
                    tmp = list(map(lambda x: np.eye(4)[int(x)],notes)) #to_categorical hack 
                    if flipLR:
                        tmp = [tmp[3],tmp[1],tmp[2],tmp[0]]
                    if flipUD:
                        tmp = [tmp[0],tmp[2],tmp[1],tmp[3]]
                    #https://stackoverflow.com/questions/49684379/numpy-equivalent-to-keras-function-utils-to-categorical
                    output.append(tmp)
                    mnd_data.append((time_point, note, start_long, end_long))
                    last_time = time_point
                time_point += time_resolution
            stored_lines = []
        else:
            if len(line) == 4:
                stored_lines.append(line)
        if ";" in line:
            return (mnd_data, output)
#Generator for all of the data involved in a single run of the network
def generate_song_inout_data(data_tuple):
    (bpm_data, offset, position, dirpath, chart_path, music_path) = data_tuple
    bpm_list = process_bpm(bpm_data)
    with open(chart_path, encoding="latin-1") as chart:
        chart.seek(position)
        (mnd_arr, out_arr) = mnd_getdata(chart)
    assert(len(mnd_arr) == len(out_arr))
    assert(len(mnd_arr) > 10)
    
    weight = 20/math.sqrt(max(len(mnd_arr),100)) #lengths vary from ~100 low, ~300 average, ~1500 max
    inputs = []
    fulldata = []
    blank_input = np.zeros(25) #9 mnd data, 16 step history
    for i in range(LSTM_HISTORY_LENGTH):
        inputs.append(blank_input)

    bpm_id = 0
    bpm = bpm_list[bpm_id][1]
    next_bpm_time = bpm_list[bpm_id+1][0]*48
    
    now_sec = -offset
    next_sec = now_sec
    next_beat = mnd_arr[0][0]
    now_beat = 0
    
    for i in range(len(bpm_list)-1):
        assert(bpm_list[i][1] > 0)
        if bpm_list[i][0] > bpm_list[i+1][0]:
            print(chart_path,bpm_data,bpm_list)
            print(i,bpm_list[i][0],bpm_list[i+1][0])
        assert(bpm_list[i][0] <= bpm_list[i+1][0])
    assert(bpm_list[0][0] == 0.0)
    minilast_beat = 0
    while next_beat > next_bpm_time:
        assert(minilast_beat <= next_bpm_time)
        next_sec += (next_bpm_time-minilast_beat)/(bpm/60*48)
        minilast_beat = next_bpm_time
        bpm_id += 1
        bpm = bpm_list[bpm_id][1]
        next_bpm_time = bpm_list[bpm_id+1][0]*48
    next_sec = (next_beat-minilast_beat)/(bpm/60*48)

    for pos in range(len(mnd_arr)):
        last_beat = now_beat
        i = pos+LSTM_HISTORY_LENGTH
        (now_beat, note, start_long, end_long) = mnd_arr[pos]
        assert(now_beat == next_beat)
        minilast_beat = now_beat
        last_sec = now_sec
        now_sec = next_sec
        next_beat = mnd_arr[pos+1][0] if pos+1 < len(mnd_arr) else now_beat+(192*5)
        while next_beat > next_bpm_time:
            assert(minilast_beat <= next_bpm_time)
            next_sec += (next_bpm_time-minilast_beat)/(bpm/60*48)
            minilast_beat = next_bpm_time
            bpm_id += 1
            bpm = bpm_list[bpm_id][1]
            next_bpm_time = bpm_list[bpm_id+1][0]*48
        next_sec += (next_beat-minilast_beat)/(bpm/60*48)
        beat_fract = beat_find(now_beat)
        assert(next_sec > now_sec)
        mnd_data = [min(now_sec-last_sec,2)/2, min(next_sec-now_sec,2)/2,
                    min((now_beat-last_beat)/384,1), min((next_beat-now_beat)/384,1),
                    beat_fract/48, note/3, start_long/3, end_long/3, min(bpm/400,1)]
        #array: [Seconds prev to now, Seconds from now to next, beats prev to now, beats now to next,
        #      beat fraction, basic press count, freeze start count, freeze end count, bpm]
        #Plus the Left/Up/Down/Right arrows' data (one-hot x4 for input and output)
        
        input_mnd_aux = mnd_data.copy()
        mnd_data.extend(np.ravel(out_arr[pos]))
        inputs.append(np.array(mnd_data))
        in_arr = np.array(inputs[pos:i])
        out_dat = np.array(out_arr[pos])
        
        yield (input_mnd_aux, in_arr, out_dat[0],out_dat[1],out_dat[2],out_dat[3],weight)

#Take in a song_data tuple and return a full set of training data
def map_data_to_training_data(data_tuple):
    all_data = generate_song_inout_data(data_tuple)
    try:
        zipped_data = zip(*all_data)
        return zipped_data
    except ValueError:
        print("BAD DATA: ",data_tuple[3])
        return ([],[],[],[],[],[],[])
    (in_mnd, in_hist) = zip(*ins)
    (outL, outU, outD, outR) = zip(*outs)
    #disable data augmentation to get more unique songs in memory
    return (in_mnd, in_hist, outL, outU, outD, outR, weights)


def generate_dataset():
    gen = song_data()
    dataset = []
    try:
        while True:
            dataset.append(next(gen))
    except StopIteration:
        pass
    return dataset

def huge_full_dataset():
    #load dataset from disk (chart list)
    if os.path.exists("ddr_dataset.p"):
        print("loading dataset")
        dataset = pickle.load(open("ddr_dataset.p","rb"))
        print("dataset loaded!")
    else:
        dataset = generate_dataset()
        pickle.dump(dataset, open("ddr_dataset.p","wb"))
    bag = dataset#.copy() : Don't need to copy when doing shuffle+slice rather than shuffle+pop
    song_count = len(dataset)
    max_iters = song_count//songs_per
    excess = song_count%songs_per
    #Creates "super-batches" of random songs due to memory limitations
    with Pool(processes=7) as pool:
        while True:
            print("reloading dataset bag: size="+str(song_count))
            random.shuffle(bag)
            next_dataSlice = bag[0:excess] #First set is smaller
            next_data_superbatch = pool.imap_unordered(map_data_to_training_data,next_dataSlice)
            for x in range(max_iters+1):
                #if not next_data_superbatch.ready():
                #    print(">>>DATA NOT READY!")
                #data_superbatch_complete = next_data_superbatch.get()
                in_mnd_set = []
                in_hist_set = []
                outL_set = []
                outU_set = []
                outD_set = []
                outR_set = []
                weights_set = []
                for data in next_data_superbatch:
                    (in_mnd, in_hist, outL, outU, outD, outR, weights) = data
                    in_mnd_set.extend(in_mnd)
                    in_hist_set.extend(in_hist)
                    #Data augmentation check
                    #m = len(in_hist)//4
                    #for x in range(64,80):
                    #    for i in range(4):
                    #      print(in_hist[x+i*m][31][7:11])
                    #print("----")
                    outL_set.extend(outL)
                    outU_set.extend(outU)
                    outD_set.extend(outD)
                    outR_set.extend(outR)
                    weights_set.extend(weights)
                    
                print("Data prepared (",x*songs_per+excess,"/",song_count,")")
                in_mnd_set = np.array(in_mnd_set)
                in_hist_set = np.array(in_hist_set)
                outL_set = np.array(outL_set)
                outU_set = np.array(outU_set)
                outD_set = np.array(outD_set)
                outR_set = np.array(outR_set)
                weights_set = np.array(weights_set)
                print("Data sent")
                yield ((in_mnd_set,in_hist_set),
                    (outL_set,outU_set,outD_set,outR_set),weights_set)
                if x < max_iters:
                    next_dataSlice = bag[x*songs_per+excess:(x+1)*songs_per+excess]
                    next_data_superbatch = pool.imap_unordered(map_data_to_training_data,next_dataSlice)



if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import LambdaCallback
    from tensorflow.keras import layers
    from tensorflow.keras import models
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.models import load_model
    model_make = True
    if len(sys.argv) > 1:
        if sys.argv[1] == "LOAD":
            print("Loading model")
            model = load_model("ddr_model.h5")
            #If loading from disk, maybe assume the model is reasonably-trained and use the slower but "better" SGD algorithm?
            optimizer = keras.optimizers.Nadam()
            if len(sys.argv) > 2:
                if sys.argv[2] == "SGD":
                    optimizer = keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True)
                if sys.argv[2] == "RMS":
                    optimizer = RMSprop(learning_rate=0.0002)
                if sys.argv[2] == "KEEP":
                    optimizer = None
            if optimizer is not None:
                model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            model_make = False
    if model_make:
        hist_input = layers.Input(shape=(LSTM_HISTORY_LENGTH,25,),name="hist_input")
        hist_lstm = layers.TimeDistributed(layers.Dense(64, activation='elu'))(hist_input)
        hist_lstm = layers.TimeDistributed(layers.Dense(64, activation='elu'))(hist_lstm)
        hist_lstma = layers.LSTM(256,return_sequences=True)(hist_lstm)
        hist_lstmb = layers.LSTM(64,return_sequences=True,go_backwards=True)(hist_lstm)
        hist_lstm = layers.concatenate([hist_lstma,hist_lstmb])
        hist_lstm = layers.LSTM(256)(hist_lstm)
        mnd_input = layers.Input(shape=(9,),name="mnd_input")
        x = layers.Dense(32, activation='elu')(mnd_input)
        x = layers.Dense(32, activation='elu')(x)
        x = layers.concatenate([x,hist_lstm])
        x = layers.Dense(400, activation='elu')(x)
        x = layers.Dense(320, activation='elu')(x)
        x = layers.Dense(256, activation='elu')(x)
        x = layers.Dense(256, activation='elu')(x)
        x = layers.Dense(64, activation='elu')(x)
        outL = layers.Dense(4, activation='softmax', name = "outL")(x)
        outU = layers.Dense(4, activation='softmax', name = "outU")(x)
        outD = layers.Dense(4, activation='softmax', name = "outD")(x)
        outR = layers.Dense(4, activation='softmax', name = "outR")(x)

        optimizer = keras.optimizers.Nadam()
        model = models.Model(inputs=[mnd_input,hist_input],outputs=[outL,outU,outD,outR])

        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()
    huge_gen = huge_full_dataset()
    while True: #huge_full_dataset will keep repeating after going through all songs
        (ins, outs, weights) = next(huge_gen)
        model.fit(ins,outs,batch_size=512)
        del ins
        del outs
        del weights
        print("Saving model")
        if os.path.exists("ddr_modelBACKUP.h5"):
            os.remove("ddr_modelBACKUP.h5")
        if os.path.exists("ddr_model.h5"):
            os.rename("ddr_model.h5","ddr_modelBACKUP.h5")
        model.save("ddr_model.h5")
        print("Save complete")
