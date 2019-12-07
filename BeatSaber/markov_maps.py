import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import soundfile as sf
import csv
import pickle as pkl

def main2(directory):
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    for file in files:
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
                #lights = sample['lights']
                #obstacles = sample['obstacles'] # to be added in the future
                for time in range(int(songlength)):
                    window = data[samplerate*time:samplerate*(time+1)]
                    beatrate = 147
                    ## Old Code, too high dimensional
                    # label = np.zeros([beatrate+1, 5, 3, 2, 5])
                    # for entry in notes:
                    #     if entry['_time'] >= time and entry['_time'] < time + 1 and entry['_type'] < 3:
                    #         # bandied fix for the weird omnidirectional boxes (8) - don't want a massive tensor, could set to index 4
                    #         if (entry['_cutDirection']) > 3:
                    #             entry['_cutDirection'] = 4
                    #         #print(np.floor((entry['_time'] % 1) * beatrate).astype(int), ((entry['_time'] % 1) * beatrate))
                    #         label[np.floor((entry['_time'] % 1) * beatrate).astype(int), entry['_lineIndex'], entry['_lineLayer'], entry['_type'], entry['_cutDirection']] = 1
                    #         with open('./samples_window/' + file.replace('.pkl','').replace('./samples/','') + str(time) + '.pkl', 'wb') as f2:
                    #             pkl.dump({'name':name, 'time':entry['_time'], 'window':window, 'label':label}, f2)
                    #label = np.zeros([beatrate+1])
                    label = []
                    for entry in notes:
                        if entry['_time'] >= time and entry['_time'] < time + 1 and entry['_type'] < 3:
                            # bandied fix for the weird omnidirectional boxes (8) - don't want a massive tensor, could set to index 4
                            if (entry['_cutDirection']) > 3:
                                entry['_cutDirection'] = 4
                            #print(np.floor((entry['_time'] % 1) * beatrate).astype(int), ((entry['_time'] % 1) * beatrate))
                            #label[np.floor((entry['_time'] % 1) * beatrate).astype(int)] = 1
                            # TODO : somehow encode this into a letter in a way that can be decoded
                            letter = [entry['_lineIndex'], entry['_lineLayer'], entry['_type'], entry['_cutDirection']]
                            label.append(letter)
                            with open('./markov/' + file.replace('.pkl','').replace('./samples/','') + str(time) + '.pkl', 'wb') as f2:
                                pkl.dump({'mapping':label}, f2)
                    

if __name__ == "__main__":
    directory = './samples/'
    main2(directory)