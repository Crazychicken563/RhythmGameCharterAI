import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import soundfile as sf
import csv
import pickle as pkl
from aubio import source, onset
import markovify

def main2(directory):
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    #beat_samples = [join('./beat_samples', f) for f in listdir('./beat_samples') if isfile(join('./beat_samples', f))]
    for file in files:
        print(file)
        try:
            with open(file, 'rb') as f:
                ## NOTE For reconstruction, this data needs to be added to *items*
                #sample = pkl.load(f)
                name = file # this is needed for reconstruction of the song

                s = source(file)
                data, samplerate = sf.read(file)
                #samplerate = sample['samplerate']
                songlength = data.shape[0]/samplerate

                win_s = 256                 # fft size
                hop_s = win_s // 2          # hop size
                #o = onset("default", win_s, hop_s, samplerate)
                o = onset('specdiff', 512*8, 512, samplerate=samplerate)
                # list of onsets, in samples
                onsets = []

                # total number of frames read
                total_frames = 0
                while True:
                    samples, read = s()
                    if o(samples):
                        if(o.get_last_s() >= songlength - 2): # something is wrong w this api, so lets kill the onsets manually
                            break
                        print('Beat Detected:', o.get_last_s())
                        onsets.append(o.get_last())
                    total_frames += read

                # yeah not dealing with 48000 right now
                assert samplerate == 44100

                print(len(onsets))

                with open('markov.txt', 'r') as f2:
                    sentences = f2.readline()

                text_model = markovify.Text(sentences)

                sentence = 'hi there'

                while(len(sentence.split(' ')) < len(onsets)):
                    sentence = text_model.make_sentence(max_words=len(onsets)*3)

                mapping = sentence.split(' ')

                print(name, samplerate)

                with open('./output/Normal.dat', 'r') as f6:
                    line = f6.readline()
                    choreo = eval(line)
                    choreo['_obstacles'] = [] # set the obstacles to empty, since we didnt predict those
                    choreo['_notes'] = [] # clear both the notes and events
                    choreo['_events'] = []

                for idx, entry in enumerate(onsets):
                    mapp = mapping[idx]
                    time = entry/samplerate
                    cutDirection = int(mapp[1])
                    lineLayer = int(mapp[3])
                    lineIndex = int(mapp[2])
                    type = int(mapp[0])
                    note = {'_cutDirection': cutDirection, '_lineIndex': lineIndex, '_lineLayer': lineLayer, '_time': time, '_type': type}
                    # we're just going to randomly assign lighting
                    event = {'_time': time, '_type': type, '_value': cutDirection}
                    choreo['_notes'].append(note)
                    choreo['_events'].append(event)
            
            with open('./output/Normal.dat', 'w') as f4:
                f4.write(repr(choreo).replace('\'', '\"').replace(' ', ''))
                
        except Exception as f:
            print(f)

if __name__ == "__main__":
    directory = './samples_infer/'
    #main(directory)
    main2(directory) #second preprocessing needed 