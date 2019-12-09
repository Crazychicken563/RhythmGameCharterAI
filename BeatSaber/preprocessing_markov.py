import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import soundfile as sf
import csv
import pickle as pkl
import markovify

def main2(directory):
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    #beat_samples = [join('./beat_samples', f) for f in listdir('./beat_samples') if isfile(join('./beat_samples', f))]
    sentence = ''
    for file in files:
        print(file)
        try:
            with open(file, 'rb') as f:
                ## NOTE For reconstruction, this data needs to be added to *items*
                sample = pkl.load(f)
                name = sample['name'] # this is needed for reconstruction of the song
                data = sample['songdata']
                samplerate = sample['samplerate']
                songlength = data.shape[0]/samplerate
                # yeah not dealing with 48000 right now
                if samplerate == 44100:
                    notes = sample['notes']
                    print(name, samplerate)
                    #beatrate = 441
                    for entry in notes:
                        #mapping = np.zeros([int(np.ceil(songlength*beatrate))])
                        time = entry['_time']/float(sample['BMP']) * 60 #static "60" BPM to match up to music
                        #songwindow = data[int((time-0.1)*samplerate):int(time*samplerate),:]
                        songwindow = data[int(time*samplerate)-int(samplerate/10):int(time*samplerate),:]
                        cutDirection = entry['_cutDirection'] #0-8
                        lineIndex = entry['_lineIndex'] #0-3
                        lineLayer = entry['_lineLayer'] #0-2
                        noteType = entry['_type'] #0-1 no bombs please -> remove 3 
                        cutDirection = entry['_cutDirection']
                        sentence += ('{}{}{}{} '.format(noteType,cutDirection,lineIndex,lineLayer))
                        #print(int(np.round(time*beatrate)))
                        #if noteType <= 1:
                        #    label = np.zeros([2,3,4,9])
                        #    print(noteType, lineLayer, lineIndex, cutDirection)
                        #    label[noteType, lineLayer, lineIndex, cutDirection] = 1
                        #    #mapping[int(np.round(time*beatrate))] = 1 
                        #    with open('./beat_samples/' + file.replace('.pkl','').replace('./samples/','') + str(time) + '.pkl', 'wb') as f2:
                        #        pkl.dump({'name':name, 'time':time, 'window':songwindow, 'label':label}, f2)
                    #with open('./markov/' + file.replace(directory,''), 'wb') as f3:
                    #    pkl.dump(sentence[:-1], f3)
                    sentence = sentence[:-1] + '. '
        except Exception as f:
            print(f)

    with open('markov.txt', 'w') as f:
        f.write(sentence[:-1])
        #pkl.dump(sentence[:-1], f)

if __name__ == "__main__":
    directory = './samples/'
    #main(directory)
    main2(directory) #second preprocessing needed 