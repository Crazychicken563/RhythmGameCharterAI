import librosa
import numpy as np
import pickle
import os
np.set_printoptions(precision=4,suppress=True, floatmode="fixed", sign=" ")
SAMPLE_RATE = 44100
SPECT_SKIP = 128
AUDIO_BEFORE_LEN = SAMPLE_RATE//SPECT_SKIP
AUDIO_AFTER_LEN = SAMPLE_RATE//SPECT_SKIP//4
NOTE_HISTORY = 32
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

def load_constq_spectogram(data_file,songfile):
    cache_spect = os.path.join(os.path.dirname(data_file),"44khz_constqspecto_128.p")
    if os.path.exists(cache_spect):
        return pickle.load(open(cache_spect,"rb"))
    else:
        raw_audio = load_song(data_file, songfile)
        spectogram = librosa.core.cqt(raw_audio, sr=SAMPLE_RATE, hop_length=SPECT_SKIP)
        spectogram = librosa.amplitude_to_db(np.abs(spectogram),ref=np.max)
        pickle.dump(spectogram, open(cache_spect,"wb"))
        return spectogram

def onset_strengths(data_file, songfile):
    cache_onset = os.path.join(os.path.dirname(data_file),"onsets_44khz.p")
    if os.path.exists(cache_onset):
        onsets_base = pickle.load(open(cache_onset,"rb"))
    else:
        spect = load_spectogram(data_file,songfile)
        onsets_base = librosa.onset.onset_strength(S=spect)
        max_n = onsets_base.max()
        min_n = onsets_base.min()
        onsets_base = (onsets_base-min_n)/(max_n-min_n)
        pickle.dump(onsets_base, open(cache_onset,"wb"))

    cache_onset_cqt = os.path.join(os.path.dirname(data_file),"onsets_cqt_44khz.p")
    if os.path.exists(cache_onset_cqt):
        onsets_cqt = pickle.load(open(cache_onset_cqt,"rb"))
    else:
        spect_cqt = load_constq_spectogram(data_file,songfile)
        onsets_cqt = librosa.onset.onset_strength(S=spect_cqt)
        max_n = onsets_cqt.max()
        min_n = onsets_cqt.min()
        onsets_cqt = (onsets_cqt-min_n)/(max_n-min_n)
        pickle.dump(onsets_cqt, open(cache_onset_cqt,"wb"))

    cache_rms = os.path.join(os.path.dirname(data_file),"rms_44khz.p")
    if os.path.exists(cache_rms):
        rms_dat = pickle.load(open(cache_rms,"rb"))
    else:
        spect = load_constq_spectogram(data_file,songfile)
        rms_dat = librosa.feature.rms(S=spect).flatten()
        max_n = rms_dat.max()
        min_n = rms_dat.min()
        rms_dat = (rms_dat-min_n)/(max_n-min_n)
        pickle.dump(rms_dat, open(cache_rms,"wb"))

    cache_zero_rate = os.path.join(os.path.dirname(data_file),"zero_rate_44khz.p")
    if os.path.exists(cache_zero_rate):
        zero_rate = pickle.load(open(cache_zero_rate,"rb"))
    else:
        song = load_song(data_file,songfile)
        zero_rate = librosa.feature.zero_crossing_rate(y=song,hop_length=SPECT_SKIP).flatten()
        pickle.dump(zero_rate, open(cache_zero_rate,"wb"))

    base_len = onsets_base.shape[0]
    cqt_len = onsets_cqt.shape[0]
    rms_len = rms_dat.shape[0]
    zcr_len = zero_rate.shape[0]
    max_len = max([base_len,cqt_len,rms_len,zcr_len])
    
    
    onsets_base = np.pad(onsets_base,((0,max_len-base_len)))
    onsets_cqt = np.pad(onsets_cqt,((0,max_len-cqt_len)))
    rms_dat = np.pad(rms_dat,((0,max_len-rms_len)))
    zero_rate = np.pad(zero_rate,((0,max_len-zcr_len)))
    
    if (onsets_base.shape != onsets_cqt.shape or onsets_cqt.shape != rms_dat.shape or rms_dat.shape != zero_rate.shape):
        print(songfile)
        print(onsets_base.shape,onsets_cqt.shape,rms_dat.shape,zero_rate.shape)
    
    return np.stack((onsets_base,onsets_cqt,rms_dat,zero_rate),axis=-1)
    
def bpm_split(text):
    x = text.split("=")
    assert (len(x) == 2)
    return (float(x[0]),float(x[1]))
def process_bpm(data_in):
  bpm_text = data_in.split(",")
  bpm_list = list(map(bpm_split,bpm_text))
  bpm_list.append((float("inf"),0))
  return bpm_list