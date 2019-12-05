
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

time_resolution = 48
note_freq = 3
jump_freq = .08
long_freq = .02
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

raw_audio = load_song_cache(path,music_path)

audio_length = (len(raw_audio)/SAMPLE_RATE)-(PADDING*2) #added 2 sec of padding
bpm_list = process_bpm(bpm_data)


bpm_id = 0
bpm = bpm_list[bpm_id][1]
next_bpm_time = bpm_list[bpm_id+1][0]*48

now_sec = -float(offset)
last_beat = -(192*5)
now_beat = 0
mnd_id = 0
full_hist = [np.zeros(12)]*NOTE_HISTORY#9 stats data, 3 output

current_holds = 0
last_found_beat = 0
last_found_sec = 0

blank_audio = np.zeros((1,AUDIO_BEFORE_LEN+AUDIO_AFTER_LEN,1))

with open("output.mnd", mode='w', encoding="utf-8") as out:
    out.write(music_path+"\n")
    out.write(bpm_data+"\n")
    out.write(offset+"\n")
    while now_sec < audio_length+1:
        #history = stats+output (+time apart) for previous 64 notes
        #const stats = (max concurrent, any note frequency, jump freq, long freq)
        #varying stats = (bpm, time apart sec, time apart beats, current holds, fractional beat)
        t_res = beat_find(now_beat)
        stats = (max_simultaneous/3,note_freq/16,jump_freq,long_freq,
                bpm/400,now_sec-last_found_sec,min((now_beat-last_found_beat)/192,2),current_holds,t_res)
        
        id_now = sec_to_id(now_sec)
        audio = raw_audio[id_now-AUDIO_BEFORE_LEN:id_now+AUDIO_AFTER_LEN]
        audio_in = audio.reshape((1,AUDIO_BEFORE_LEN+AUDIO_AFTER_LEN,1))
        history_in = np.array(full_hist[mnd_id:mnd_id+NOTE_HISTORY],dtype="float32",ndmin=3)
        stats_in = np.array(stats,dtype="float32",ndmin=2)
        probs = model.predict((audio_in,history_in,stats_in))
        fake_probs = model.predict((blank_audio,history_in,stats_in))
        (note,start_long,end_long) = (x.astype('float') for x in probs[0])
        note       = math.floor((random.betavariate(3,3)*.6+.2)+note)
        start_long = math.floor((random.betavariate(3,3)*.6+.2)+start_long)
        end_long   = math.floor((random.betavariate(3,3)*.6+.2)+end_long)
        out_dat = (note, start_long, end_long)
        print(out_dat,probs,fake_probs)
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
            extended_stats.extend(out_dat)
            full_hist.append(extended_stats)
            current_holds += out_dat[1] - out_dat[2]
            last_found_beat = now_beat
            last_found_sec = now_sec
            #print(now_sec,now_beat,extended_stats)
        
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