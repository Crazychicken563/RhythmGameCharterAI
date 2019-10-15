# This program requires the pydub module
try:
    import pydub
except ImportError:
    import pip
    pip.main(['install', '--user', 'pydub'])
    import pydub

from pydub import AudioSegment
from pydub import effects



def load_audio(filename: str) -> (AudioSegment, str):
    _, ext = filename.split('.')
    audio = AudioSegment.from_file(filename, ext)
    return audio, ext

def normalize_volume(audio: AudioSegment) -> AudioSegment:
    song_normalized = effects.normalize(AudioSegment)
    return song_normalized


def remove_noise(audio: AudioSegment) -> AudioSegment:
    return audio


def calc_bpm(audio: AudioSegment) -> AudioSegment:
    return audio


def preprocessed_audio(filename: str, path_to_output: str, output_format: str):
    '''
    preprocess the given audio file and export to a target location
    '''
    audio, ext = load_audio(filename)
    audio = normalize_volume(audio) # Only normalization for now
    audio.export(path_to_output, format=output_format)
