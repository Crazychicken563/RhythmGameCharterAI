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
    min_silence = int(60000/240.0) # Allow up to 240 bpm
    nonsilent_ranges = detect_nonsilent(tmp_audio, min_silence, beat_volume)
    spaces_between_beats = []
    last_t = nonsilent_ranges[0][0]

    for peak_start, _ in nonsilent_ranges[1:]:
        spaces_between_beats.append(peak_start - last_t)
        last_t = peak_start

    spaces_between_beats = sorted(spaces_between_beats)
    temp = len(spaces_between_beats) / 2
    temp = int(temp)
    print(temp)
    if temp == 0:
        # This just means that this segment had no audio louder then 120 Hz
        # might as well discard it since we aren't going to get a good bpm
        # measurment from quieter sections of the song
        return 0
    space = spaces_between_beats[temp]
    bpm = 60000 / space
    return bpm

# What do we know about music and BPM?
# BPM doesnt constantly vary. It may change throughout a song,
# but it will usually snap to a different value (we can
# disragard funky cases where it gradually increases)
# This means that there should be subsections of the audio
# that have a consistent BPM value.
# We can use a sliding window to analize the bpm throughout the song
# consistent bpm values with a low error margin will indicate a copnstant BPM
# if the BPM suddenly changes we can assume that the end of the sliding window
# has hit a BPM change. We note this timestamp and later when the start of the
# sliding window reaches this timestamp we can check if the bpm is different
# the previous values. This way we detect BPM changes.

def preprocessed_audio(filename: str):
    '''
    preprocess the given audio file and export to a target location
    '''
    print("load audio at " + filename)
    audio = AudioSegment.from_wav(filename)
    #audio, ext = load_audio(filename)
    #audio = normalize_volume(audio) # Only normalization for now
    overallBPM = calc_bpm(audio)
    print("Full song bpm: " + str(overallBPM))
    windowSize = 30000 # 10 seconds
    for i in range(0, len(audio) - windowSize, 5000):
        bpmForWindow = calc_bpm(audio[i:i+windowSize])
        print("BPM from " + str(i) + " -> " + str(i+windowSize) + " = " + str(bpmForWindow))
    # audio.export('preprocessed_'+filename, format=ext)

preprocessed_audio("C:/Users/Seva/workspace/RhythmGameCharterAI/preprocessing/song.wav")