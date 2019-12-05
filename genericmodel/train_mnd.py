import os
import re

import random
import sys
import io
import pickle
import math
from generic_processing import *
import multiprocessing
from multiprocessing import Pool
#Note: Requires both ddr_finder and ddr_to_generic to be run first
#will dynamically construct both the MND's data and the output to compare against
#Current input is the previous arrows' positions/type (basic arrow, freeze start, freeze end, none)x4, MND data for both past arrows and the new step, and a very small amount of audio data around the new step
#start of song data has a ton of 0 (except for BPM)
source_dir = "../preprocessing/ddr_data"
songs_per = 2

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
                        yield this_difficulty_file


#Generator for all of the data involved in a single run of the network
def generate_song_inout_data(data_file):
    mnd_raw = []
    print(data_file)
    with open(data_file, encoding="latin-1") as chart:
        song_path = chart.readline().strip()
        bpm_data = chart.readline().strip()
        offset = float(chart.readline())
        time_resolution = 48
        max_simultaneous = 1
        long_count = 0
        any_note_count = 0
        jump_count = 0
        current_holds = 0
        for line in chart:
            line_data = tuple(int(x) for x in line.split())
            t_res = beat_find(line_data[0])
            time_resolution = math.gcd(time_resolution,t_res)
            current_holds += line_data[2]
            any_note_count += min(line_data[1]+line_data[2],1)
            long_count += line_data[2]
            if line_data[1]+line_data[2] >= 2:
                jump_count += 1
            max_simultaneous = max(max_simultaneous,line_data[1]+current_holds)
            current_holds -= line_data[3]
            mnd_raw.append(line_data)
        assert(current_holds == 0)
    mnd_raw.append((0xFFFFFFFF,0,0,0))
    
    dirpath = os.path.dirname(data_file)
    songdata = os.path.join(dirpath,"22khz_resample_pad.p")
    if os.path.exists(songdata):
        raw_audio = pickle.load(open(songdata,"rb"))
    else:
        raw_audio = load_song(song_path)
        pickle.dump(raw_audio, open(songdata,"wb"))
    
    audio_length = (len(raw_audio)/SAMPLE_RATE)-(PADDING*2) #added 2 sec of padding
    note_freq = any_note_count/audio_length
    jump_freq = jump_count/any_note_count
    long_freq = long_count/any_note_count
    bpm_list = process_bpm(bpm_data)
    
    

    bpm_id = 0
    bpm = bpm_list[bpm_id][1]
    next_bpm_time = bpm_list[bpm_id+1][0]*48
    
    now_sec = -offset
    last_beat = -(192*5)
    now_beat = 0
    mnd_id = 0
    full_hist = [np.zeros(12)]*NOTE_HISTORY#9 stats data, 3 output
  
    current_holds = 0
    last_found_beat = 0
    last_found_sec = 0
    
    while mnd_id < len(mnd_raw)-1 or now_sec < audio_length:
        #history = stats+output (+time apart) for previous 64 notes
        #const stats = (max concurrent, any note frequency, jump freq, long freq)
        #varying stats = (bpm, time apart sec, time apart beats, current holds, fractional beat)
        t_res = beat_find(now_beat)
        stats = (max_simultaneous/3,note_freq/16,jump_freq,long_freq,
                bpm/400,now_sec-last_found_sec,min((now_beat-last_found_beat)/192,2),current_holds,t_res)
        history = full_hist[mnd_id:mnd_id+NOTE_HISTORY]
        hit = False
        if mnd_raw[mnd_id][0] == now_beat:
            hit = True
            out_dat = mnd_raw[mnd_id][1:]
            mnd_id += 1
            extended_stats = list(stats)
            extended_stats.extend(out_dat)
            full_hist.append(extended_stats)
            current_holds += out_dat[1] - out_dat[2]
            last_found_beat = now_beat
            last_found_sec = now_sec
            #print(now_sec,now_beat,extended_stats)
        else:
            if not now_beat < mnd_raw[mnd_id][0]:
                print("now:"+str(now_beat)+", goal:"+str(mnd_raw[mnd_id][0])+",bpmnext:"+str(next_bpm_time))
            assert(now_beat < mnd_raw[mnd_id][0])
            out_dat = (0,0,0)
        if (hit or random.random() > 0.8/time_resolution):
            id_now = sec_to_id(now_sec)
            audio_after = raw_audio[id_now:id_now+AUDIO_AFTER_LEN]
            if len(audio_after) != AUDIO_AFTER_LEN:
                print(data_file)
                print("after",now_sec,now_beat,id_now,len(raw_audio))
            audio_after = audio_after.reshape((AUDIO_AFTER_LEN,1))
            
            audio_before = raw_audio[id_now-AUDIO_BEFORE_LEN:id_now]
            if len(audio_before) != AUDIO_BEFORE_LEN:
                print(data_file)
                print("before",now_sec,now_beat,id_now,len(raw_audio))
            audio_before = audio_before.reshape((AUDIO_BEFORE_LEN,1))
            yield (audio_before, audio_after, history,stats,out_dat)
        
        last_beat = now_beat
        now_beat += time_resolution
        minilast_beat = last_beat
        while now_beat > next_bpm_time:
            assert(minilast_beat <= next_bpm_time)
            now_sec += (next_bpm_time-minilast_beat)/(bpm/60*48)
            minilast_beat = next_bpm_time
            bpm_id += 1
            bpm = bpm_list[bpm_id][1]
            next_bpm_time = bpm_list[bpm_id+1][0]*48
        now_sec += (now_beat-minilast_beat)/(bpm/60*48)

#Take in a song_data tuple and return a full set of training data
def map_data_to_training_data(data_file):
    all_data = generate_song_inout_data(data_file)
    #try:
    zipped_data = zip(*all_data)
    return zipped_data
    #except ValueError:
        #print("BAD DATA: ",data_file)
        #return ([],[],[],[],[])


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
    if os.path.exists("ddr_only_songs.p"):
        print("loading dataset")
        dataset = pickle.load(open("ddr_only_songs.p","rb"))
        print("dataset loaded!")
    else:
        dataset = generate_dataset()
        pickle.dump(dataset, open("ddr_only_songs.p","wb"))
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
                audio_before_set  = []
                audio_after_set   = []
                history_set       = []
                stats_set         = []
                out_set           = []
                for data in next_data_superbatch:
                    (audio_before, audio_after, history, stats, out) = data
                    audio_before_set.extend(audio_before)
                    audio_after_set.extend(audio_after)
                    history_set.extend(history)
                    stats_set.extend(stats)
                    out_set.extend(out)
                    
                print("Data prepared (",x*songs_per+excess,"/",song_count,")")
                audio_before_set=np.array(audio_before_set)
                audio_after_set =np.array(audio_after_set )
                history_set     =np.array(history_set     )
                stats_set       =np.array(stats_set       )
                out_set         =np.array(out_set         )
                print("Data sent")
                yield ((audio_before_set,audio_after_set,history_set,stats_set),
                    (out_set))
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
            model = load_model("mnd_model.h5")
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
        audio_before_in = layers.Input(shape=(AUDIO_BEFORE_LEN,1,),name="audio_before_in")
        #audio_b = layers.LSTM(64,return_sequences=True)(audio_before_in)
        #audio_b = layers.LSTM(64)(audio_b)
        
        audio_after_in = layers.Input(shape=(AUDIO_AFTER_LEN,1,),name="audio_after_in")
        #audio_a = layers.LSTM(64,return_sequences=True,go_backwards=True)(audio_after_in)
        #audio_a = layers.LSTM(64,go_backwards=True)(audio_a)
        
        hist_input = layers.Input(shape=(NOTE_HISTORY,12,),name="hist_input")
        hist_lstm = layers.TimeDistributed(layers.Dense(32, activation='elu'))(hist_input)
        hist_lstm = layers.TimeDistributed(layers.Dense(64, activation='elu'))(hist_lstm)
        hist_lstma = layers.LSTM(64,return_sequences=True)(hist_lstm)
        hist_lstmb = layers.LSTM(16,go_backwards=True,return_sequences=True)(hist_lstm)
        hist_lstm = layers.concatenate([hist_lstma,hist_lstmb])
        hist_lstm = layers.LSTM(128)(hist_lstm)
        
        stats_input = layers.Input(shape=(9,),name="stats_input")
        x = layers.Dense(32, activation='elu')(stats_input)
        x = layers.Dense(64, activation='elu')(x)
        x = layers.concatenate([x,hist_lstm,audio_a,audio_b])
        x = layers.Dense(768, activation='elu')(x)
        x = layers.Dense(512, activation='elu')(x)
        x = layers.Dense(256, activation='elu')(x)
        x = layers.Dense(256, activation='elu')(x)
        x = layers.Dense(128, activation='elu')(x)
        x = layers.Dense(64, activation='elu')(x)
        outs = layers.Dense(3, name = "outs")(x) #Basic Notes, Start Long, End long

        optimizer = keras.optimizers.Nadam()
        model = models.Model(inputs=[audio_before_in,audio_after_in,hist_input,stats_input],outputs=[outs])
        #audio_before/after = 1/2 and 1/8 sec
        #history = stats+output (+time apart) for previous 64 notes
        #const stats = (max concurrent, any note frequency, jump freq, long freq)
        #varying stats = (bpm, time apart sec, time apart beats, current holds, fractional beat)

        model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    huge_gen = huge_full_dataset()
    while True: #huge_full_dataset will keep repeating after going through all songs
        (ins, outs) = next(huge_gen)
        model.fit(ins,outs,batch_size=4)
        del ins
        del outs
        print("Saving model")
        if os.path.exists("song_modelBACKUP.h5"):
            os.remove("song_modelBACKUP.h5")
        if os.path.exists("song_model.h5"):
            os.rename("song_model.h5","song_modelBACKUP.h5")
        model.save("song_model.h5")
        print("Save complete")