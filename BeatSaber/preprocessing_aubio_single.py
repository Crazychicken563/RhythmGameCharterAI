import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import soundfile as sf
import csv
import pickle as pkl
from aubio import source, onset

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
                o = onset('phase', 512, 512, samplerate=samplerate)
                # list of onsets, in samples
                onsets = []

                # total number of frames read
                total_frames = 0
                while True:
                    samples, read = s()
                    if o(samples):
                        if(o.get_last_s() >= songlength - 1): # something is wrong w this api, so lets kill the onsets manually
                            break
                        print('Beat Detected:', o.get_last_s())
                        onsets.append(o.get_last())
                    total_frames += read

                # yeah not dealing with 48000 right now
                assert samplerate == 44100

                print(len(onsets))

                print(name, samplerate)
                beatrate = 441
                for entry in onsets:
                    #time = entry['_time']/float(sample['BMP']) * 60 #static "60" BPM to match up to music
                    time = entry/samplerate
                    #songwindow = data[int((time-0.1)*samplerate):int(time*samplerate),:]
                    songwindow = data[int(time*samplerate)-int(samplerate/10):int(time*samplerate),:]
                    # cutDirection = entry['_cutDirection'] #0-8
                    # lineIndex = entry['_lineIndex'] #0-3
                    # lineLayer = entry['_lineLayer'] #0-2
                    # noteType = entry['_type'] #0-1 no bombs please -> remove 3 
                    # cutDirection = entry['_cutDirection']
                    #print(int(np.round(time*beatrate)))
                    label = np.zeros([2,3,4,9])
                    with open('./beat_samples_infer/' + file.replace('.pkl','').replace(directory,'') + str(time) + '.pkl', 'wb') as f2:
                        pkl.dump({'name':name, 'time':time, 'window':songwindow, 'label':label}, f2)
                
        except Exception as f:
            print(f)

if __name__ == "__main__":
    directory = './samples_infer/'
    #main(directory)
    main2(directory) #second preprocessing needed 