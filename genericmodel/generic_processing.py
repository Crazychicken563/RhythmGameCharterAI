import librosa
import numpy as np
import pickle
import os
np.set_printoptions(precision=4,suppress=True, floatmode="fixed", sign=" ")
SAMPLE_RATE = 22050
AUDIO_BEFORE_LEN = SAMPLE_RATE//4
AUDIO_AFTER_LEN = SAMPLE_RATE//16
NOTE_HISTORY = 2
PADDING = 5
def sec_to_id(seconds):
    return round((seconds+PADDING)*SAMPLE_RATE)
def beat_find(time_point):
    for tt in [48, 24, 16, 12, 8, 6, 4, 3, 2, 1]: #time_resolution of individual notes
        if time_point%tt == 0:
            return tt
    raise Exception(f"beat_frac {time_point} not int")

def load_song_cache(data_file, songfile):
    cache = os.path.join(os.path.dirname(data_file),"22khz_resample_pad.p")
    if os.path.exists(cache):
        return pickle.load(open(cache,"rb"))
    else:
        raw_audio = load_song(songfile)
        pickle.dump(raw_audio, open(cache,"wb"))
        return raw_audio

def load_song(filename):
    with librosa.warnings.catch_warnings():
        librosa.warnings.simplefilter("ignore")
        song, sr = librosa.load(filename, sr=SAMPLE_RATE, mono=True)
        assert(sr == SAMPLE_RATE)
        #print(filename)
        return np.pad(song,SAMPLE_RATE*PADDING)

def bpm_split(text):
    x = text.split("=")
    assert (len(x) == 2)
    return (float(x[0]),float(x[1]))
def process_bpm(data_in):
  bpm_text = data_in.split(",")
  bpm_list = list(map(bpm_split,bpm_text))
  bpm_list.append((float("inf"),0))
  return bpm_list