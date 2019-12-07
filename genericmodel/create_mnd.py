
from generic_processing import *
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
import math
model = load_model("song_model.h5")

time_resolution = 12
note_freq = 4.5
jump_freq = .15
long_freq = .1
max_simultaneous = 2

if len(sys.argv) > 1:
  path = sys.argv[1]
else:
  path = "../preprocessing/ddr_data/In The Groove/Anubis/chart_data.dat"


with open(path, mode='r', encoding="utf-8") as chart_data:
    (a, chart_path) = chart_data.readline().strip().split(":",1)
    (b, music_path) = chart_data.readline().strip().split(":",1)
    (c, bpm_data)        = chart_data.readline().strip().strip(";").split(":")
    (d, offset)     = chart_data.readline().strip().split(":")
    assert (a, b, c, d) == ("CHART", "MUSIC", "BPM", "OFFSET")

raw_audio = onset_strengths(path,music_path)

audio_length = (len(raw_audio)*SPECT_SKIP/SAMPLE_RATE)-(PADDING*2)
bpm_list = process_bpm(bpm_data)


bpm_id = 0
bpm = bpm_list[bpm_id][1]
next_bpm_time = bpm_list[bpm_id+1][0]*48

now_sec = -float(offset)
now_beat = 0
mnd_id = 0            

current_holds = 0
last_found_beat = -192
last_found_sec = -1

blank_audio = np.random.random((1,AUDIO_BEFORE_LEN+AUDIO_AFTER_LEN,4))

def select_prob(probs):
    best_v = 0
    best_id = 0
    for (i, p) in enumerate(probs):
        sample_v = p*random.uniform(0.6,1)
        if sample_v > best_v:
          best_id = i
          best_v = sample_v
    return best_id
    

with open("output.mnd", mode='w', encoding="utf-8") as out:
    out.write(music_path+"\n")
    out.write(bpm_data+"\n")
    out.write(offset+"\n")
    while now_sec < audio_length+4:
        #const stats = (max concurrent, any note frequency, jump freq, long freq, overall time resolution)
        #varying stats = (bpm, time apart sec (max 2), time apart beats (max 2 measures), current holds, fractional beat)
        t_res = beat_find(now_beat)
        stats = (max_simultaneous,note_freq/5,jump_freq,long_freq,time_resolution/48,
                bpm/400,min(now_sec-last_found_sec,2)/2,min((now_beat-last_found_beat)/384,1),current_holds/4,t_res/48)
        
        id_now = sec_to_id(now_sec)
        audio = raw_audio[id_now-AUDIO_BEFORE_LEN:id_now+AUDIO_AFTER_LEN]
        audio_in = np.array(audio,dtype="float32",ndmin=3)
        stats_in = np.array(stats,dtype="float32",ndmin=2)
        probs = model.predict((audio_in,stats_in))
        (note_p,start_long_p,end_long_p) = (x[0] for x in probs)
        
        note = select_prob(note_p)
        start_long = select_prob(start_long_p)
        end_long = select_prob(end_long_p)
        note -= start_long
        out_dat = (note, start_long, end_long)
        print(out_dat,note_p,start_long_p,end_long_p,now_sec)
        #print(t_res, out_dat,np.array(list(1-x[0][0] for x in probs)),np.array(list(1-x[0][0] for x in fake_probs)),np.sum(raw_audio[id_now-4:id_now+1]))
        hit = False
        for x in out_dat:
            if x > 0:
                hit = True
            if x < 0 or x > max_simultaneous:
                print("INVALID PREDICTION!!!!!")
        if hit:
            out.write(f"{now_beat} {note} {start_long} {end_long}\n")
            mnd_id += 1
            extended_stats = list(stats)
            current_holds += out_dat[1] - out_dat[2]
            last_found_beat = now_beat
            last_found_sec = now_sec
            #print(now_sec,now_beat,extended_stats)
        
        minilast_beat = now_beat
        now_beat += time_resolution
        while now_beat > next_bpm_time:
            assert(minilast_beat <= next_bpm_time)
            now_sec += (next_bpm_time-minilast_beat)/(bpm/60*48)
            minilast_beat = next_bpm_time
            bpm_id += 1
            bpm = bpm_list[bpm_id][1]
            next_bpm_time = bpm_list[bpm_id+1][0]*48
        now_sec += (now_beat-minilast_beat)/(bpm/60*48)