import librosa
import numpy as np
import pickle
import os
np.set_printoptions(precision=4,suppress=True, floatmode="fixed", sign=" ")
SAMPLE_RATE = 44100
SPECT_SKIP = 128
AUDIO_BEFORE_LEN = SAMPLE_RATE//SPECT_SKIP
AUDIO_AFTER_LEN = SAMPLE_RATE//SPECT_SKIP//4
NOTE_HISTORY = 64
PADDING = 8
def sec_to_id(seconds):
    return round((seconds+PADDING)*(SAMPLE_RATE/SPECT_SKIP))
def beat_find(time_point):
    for tt in [48, 24, 16, 12, 8, 6, 4, 3, 2, 1]: #time_resolution of individual notes
        if time_point%tt == 0:
            return tt
    raise Exception(f"beat_frac {time_point} not int")

def load_song(data_file, songfile):
    cache = os.path.join(os.path.dirname(data_file),"44khz_pad.p")
    if os.path.exists(cache):
        return pickle.load(open(cache,"rb"))
    else:
        with librosa.warnings.catch_warnings():
            librosa.warnings.simplefilter("ignore")
            song, sr = librosa.load(songfile, sr=SAMPLE_RATE, mono=True)
            assert(sr == SAMPLE_RATE)
            song = np.pad(song,SAMPLE_RATE*PADDING)
            pickle.dump(song, open(cache,"wb"))
            return song

def load_spectogram(data_file,songfile):
    cache_spect = os.path.join(os.path.dirname(data_file),"44khz_malspecto_128.p")
    if os.path.exists(cache_spect):
        return pickle.load(open(cache_spect,"rb"))
    else:
        raw_audio = load_song(data_file, songfile)
        spectogram = librosa.feature.melspectrogram(raw_audio, sr=SAMPLE_RATE, hop_length=SPECT_SKIP)
        pickle.dump(spectogram, open(cache_spect,"wb"))
        return spectogram


def onset_strength(data_file, songfile):
    cache_onset = os.path.join(os.path.dirname(data_file),"onsets_22khz.p")
    if os.path.exists(cache_onset):
        return pickle.load(open(cache_onset,"rb"))
    else:
        spect = load_spectogram(data_file,songfile)
        onsets = librosa.onset.onset_strength(S=spect)
        max = onsets.max()
        min = onsets.min()
        onsets = (onsets-min)/(max-min)
        pickle.dump(onsets, open(cache_onset,"wb"))
        return onsets
    
def bpm_split(text):
    x = text.split("=")
    assert (len(x) == 2)
    return (float(x[0]),float(x[1]))
def process_bpm(data_in):
  bpm_text = data_in.split(",")
  bpm_list = list(map(bpm_split,bpm_text))
  bpm_list.append((float("inf"),0))
  return bpm_list