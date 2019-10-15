# This program requires the pydub module
try:
    import pydub
except ImportError:
    import pip
    pip.main(['install', '--user', 'pydub'])
    import pydub

from pydub import AudioSegment
from pydub import effects
from pydub.silence import detect_nonsilent



def load_audio(filename: str) -> (AudioSegment, str):
    _, ext = filename.split('.')
    audio = AudioSegment.from_file(filename, ext)
    return audio, ext

def normalize_volume(audio: AudioSegment) -> AudioSegment:
    song_normalized = effects.normalize(AudioSegment)
    return song_normalized


def remove_noise(audio: AudioSegment) -> AudioSegment:
    return audio.get_array_of_samples


def calc_bpm(audio: AudioSegment) -> int:
    tmp_audio = effects.low_pass_filter(audio, 120) # cut off sounds below 120 Hz
    beat_volume = tmp_audio.dBFS
    min_silence = int(60000/3000) # Allow up to 3000 bpm
    nonsilent_ranges = detect_nonsilent(tmp_audio, min_silence, beat_volume)
    spaces_between_beats = []
    last_t = nonsilent_ranges[0][0]

    for peak_start, _ in nonsilent_ranges[1:]:
        spaces_between_beats.append(peak_start - last_t)
        last_t = peak_start

    spaces_between_beats = sorted(spaces_between_beats)
    space = spaces_between_beats[len(spaces_between_beats) / 2]
    bpm = 60000 / space
    return bpm


def preprocessed_audio(filename: str):
    '''
    preprocess the given audio file and export to a target location
    '''
    audio, ext = load_audio(filename)
    audio = normalize_volume(audio) # Only normalization for now
    audio.export('preprocessed_'+filename, format=ext)
