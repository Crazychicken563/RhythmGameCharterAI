import librosa
import numpy as np
NOTE_HISTORY = 4
import pickle
import os
np.set_printoptions(precision=4,suppress=True, floatmode="fixed", sign=" ")
SAMPLE_RATE = 22050
SPECT_SKIP = 128
AUDIO_BEFORE_LEN = SAMPLE_RATE//SPECT_SKIP
AUDIO_AFTER_LEN = SAMPLE_RATE//SPECT_SKIP//4
PADDING = 8
def sec_to_id(seconds):
    return round((seconds+PADDING)*(SAMPLE_RATE/SPECT_SKIP))
def beat_find(time_point):
    for tt in [48, 24, 16, 12, 8, 6, 4, 3, 2, 1]: #time_resolution of individual notes
        if time_point%tt == 0:
            return tt
    raise Exception(f"beat_frac {time_point} not int")
    
def try_pickle(cache_name):
    if os.path.exists(cache_name):
        try:
            return pickle.load(open(cache_name,"rb"))
        except:
            print("Corrupted pickle")
    return None
def cachename(data_file,primaryname,skip=True):
    if skip:
        name = primaryname+"_"+str(SAMPLE_RATE//1000)+"khz_"+str(SPECT_SKIP)+".p"
    else:
        name = primaryname+"_"+str(SAMPLE_RATE//1000)+"khz.p"
    return os.path.join(os.path.dirname(data_file),name)
    
def load_song(data_file, songfile):
    cache = cachename(data_file,"song_padded",skip=False)
    song = try_pickle(cache)
    if song is None:
        with librosa.warnings.catch_warnings():
            librosa.warnings.simplefilter("ignore")
            song, sr = librosa.load(songfile, sr=SAMPLE_RATE, mono=True)
            assert(sr == SAMPLE_RATE)
            song = np.pad(song,SAMPLE_RATE*PADDING)
            pickle.dump(song, open(cache,"wb"))
    return song

def load_constq_spectogram(data_file,songfile):
    cache_spect = cachename(data_file,"constq_spect")
    l_spec = try_pickle(cache_spect)
    if l_spec is None:
        raw_audio = load_song(data_file, songfile)
        l_spec = librosa.core.cqt(raw_audio, sr=SAMPLE_RATE, hop_length=SPECT_SKIP)
        l_spec = librosa.amplitude_to_db(np.abs(l_spec),ref=np.max)
        pickle.dump(l_spec, open(cache_spect,"wb"))
    return l_spec

def normalize_data(arr):
    ret_arr = arr.flatten()
    max_n = ret_arr.max()
    min_n = ret_arr.min()
    return (ret_arr-min_n)/(max_n-min_n)
def normalize_db(arr):
    init_arr = arr.flatten()
    ret_arr = librosa.amplitude_to_db(init_arr,ref=np.max)
    max_n = ret_arr.max()
    min_n = ret_arr.min()
    return (ret_arr-min_n)/(max_n-min_n)

def onset_strengths(data_file, songfile):
    cache_rms_stft = cachename(data_file,"rms_stft")
    rms_stft = try_pickle(cache_rms_stft)
    cache_sp_band = cachename(data_file,"sp_band")
    sp_band = try_pickle(cache_sp_band)
    cache_sp_cent = cachename(data_file,"sp_cent")
    sp_cent = try_pickle(cache_sp_cent)
    cache_sp_flat = cachename(data_file,"sp_flat")
    sp_flat = try_pickle(cache_sp_flat)
    cache_onset_mel = cachename(data_file,"onsets_mel")
    onsets_mel = try_pickle(cache_onset_mel)
    
    if any(v is None for v in (rms_stft,sp_band,sp_cent,sp_flat,onsets_mel)):
        cache_stft_complex = cachename(data_file,"stft_mag")
        stft = try_pickle(cache_stft_complex)
        if stft is None:
            raw_audio = load_song(data_file, songfile)
            stft_complex = librosa.stft(raw_audio, hop_length=SPECT_SKIP)
            stft = np.abs(stft_complex)
            pickle.dump(stft, open(cache_stft_complex,"wb"))

    if rms_stft is None:
        rms_stft = normalize_data(librosa.feature.rms(S=stft))
        pickle.dump(rms_stft, open(cache_rms_stft,"wb"))
    
    if sp_band is None:
        sp_band = normalize_db(librosa.feature.spectral_bandwidth(S=stft,sr=SAMPLE_RATE,hop_length=SPECT_SKIP))
        pickle.dump(sp_band, open(cache_sp_band,"wb"))
            
    if sp_cent is None:
        sp_cent = normalize_db(librosa.feature.spectral_centroid(S=stft,sr=SAMPLE_RATE,hop_length=SPECT_SKIP))
        pickle.dump(sp_cent, open(cache_sp_cent,"wb"))
        
    if sp_flat is None:
        sp_flat = normalize_db(librosa.feature.spectral_flatness(S=stft))
        pickle.dump(sp_flat, open(cache_sp_flat,"wb"))
    
    if onsets_mel is None:
        melspect = librosa.feature.melspectrogram(S=stft**2, sr=SAMPLE_RATE, hop_length=SPECT_SKIP)
        onsets_mel = normalize_data(librosa.onset.onset_strength(S=melspect))
        pickle.dump(onsets_mel, open(cache_onset_mel,"wb"))

    cache_onset_cqt = cachename(data_file,"onsets_cqt")
    onsets_cqt = try_pickle(cache_onset_cqt)
    if onsets_cqt is None:
        spect_cqt = load_constq_spectogram(data_file,songfile)
        onsets_cqt = normalize_data(librosa.onset.onset_strength(S=spect_cqt))
        pickle.dump(onsets_cqt, open(cache_onset_cqt,"wb"))
        
    cache_zero_rate = cachename(data_file,"zero_rate")
    zero_rate = try_pickle(cache_zero_rate)
    if zero_rate is None:
        song = load_song(data_file,songfile)
        zero_rate = librosa.feature.zero_crossing_rate(y=song,hop_length=SPECT_SKIP).flatten()
        pickle.dump(zero_rate, open(cache_zero_rate,"wb"))
    
    mel_on_len = onsets_mel.shape[0]
    cqt_on_len = onsets_cqt.shape[0]
    zcr_len = zero_rate.shape[0]
    rms_stft_len = rms_stft.shape[0]
    band_len = sp_band.shape[0]
    cent_len = sp_cent.shape[0]
    flat_len = sp_flat.shape[0]
    max_len = max([mel_on_len,cqt_on_len,rms_stft_len,band_len,cent_len,flat_len,zcr_len])
    min_len = min([mel_on_len,cqt_on_len,rms_stft_len,band_len,cent_len,flat_len,zcr_len])
    assert(max_len-min_len <= 1)
    if (max_len > min_len):
        onsets_mel= np.pad(onsets_mel, ((0,max_len-mel_on_len)))
        onsets_cqt = np.pad(onsets_cqt,((0,max_len-cqt_on_len)))
        zero_rate  = np.pad(zero_rate, ((0,max_len-zcr_len)))
        rms_stft   = np.pad(rms_stft,  ((0,max_len-rms_stft_len)))
        sp_band    = np.pad(sp_band,   ((0,max_len-band_len)))
        sp_cent    = np.pad(sp_cent,   ((0,max_len-cent_len)))
        sp_flat    = np.pad(sp_flat,   ((0,max_len-flat_len)))
    
    return np.stack((onsets_mel,onsets_cqt,zero_rate,rms_stft,sp_band,sp_cent,sp_flat),axis=-1)
    
def bpm_split(text):
    x = text.split("=")
    assert (len(x) == 2)
    return (float(x[0]),float(x[1]))
def process_bpm(data_in):
  bpm_text = data_in.split(",")
  bpm_list = list(map(bpm_split,bpm_text))
  bpm_list.append((float("inf"),0))
  return bpm_list
def song_relpath(path):
    return os.path.relpath(path,os.path.dirname(os.path.dirname(os.path.dirname(path))))