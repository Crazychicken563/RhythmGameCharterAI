import librosa
import numpy as np
LSTM_HISTORY_LENGTH = 48
SAMPLE_RATE = 44100
BIN_COUNT = 84
AUDIO_HOP_LENGTH = 512 #the default
SEC_TO_POINT = SAMPLE_RATE/AUDIO_HOP_LENGTH
AUDIO_LENGTH = 128
def beat_find(time_point):
    for tt in [48, 24, 16, 12, 8, 6, 4, 3, 2, 1]: #time_resolution of individual notes
        if time_point%tt == 0:
            return tt
    raise Exception(f"beat_frac {time_point} not int")

def load_song(filename):
    with librosa.warnings.catch_warnings():
        librosa.warnings.simplefilter("ignore")
        song, sr = librosa.load(filename, sr=SAMPLE_RATE, mono=True) #resamples to sr=22050Hz
        assert(sr == SAMPLE_RATE)
        #print(filename)
        return song

def convert_full_audio(song):
    return np.transpose(librosa.core.hybrid_cqt(song, sr=SAMPLE_RATE, hop_length=AUDIO_HOP_LENGTH, n_bins=BIN_COUNT))