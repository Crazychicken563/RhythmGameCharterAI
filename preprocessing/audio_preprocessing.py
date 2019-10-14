# This program requires the pydub module
try:
    import pydub
except ImportError:
    import pip
    pip.main(['install', '--user', 'pydub'])
    import pydub

from pydub import AudioSegment
from pydub import effects


def normalize_volume(filename: str):
    _, ext = filename.split('.')
    song = AudioSegment.from_file(filename, ext)
    song_normalized = effects.normalize(song)
    return song_normalized


def remove_noise(filename: str):
    pass


def calc_bpm(filename: str):
    pass


def preprocessed_audio(filename: str):
    return normalize_volume(filename)
