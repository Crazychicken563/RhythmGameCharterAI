import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import soundfile as sf
import csv
import pickle as pkl
from aubio import source, onset


def load_songs(path):
    songs = [f for f in listdir(path) if isdir(join(path, f))]

    #print('Songs:', songs)

    init_songs = {song:None for song in songs}

    for song in songs:
        files = [join(path,song,f) for f in listdir(join(path,song)) if isfile(join(path,song,f))]
        #print(song, files)
        #attributes = ['info', 'egg', 'maps', 'cover'] # these are to be added later
        attributes = ['egg', 'maps']
        info = {attribute:None for attribute in attributes}
        info['maps'] = []
        for file in files:
            lower = file.lower()
            if '.egg' in lower or '.ogg' in lower:
                info['egg'] = file
            elif '.dat' in lower:
                if 'info' in lower:
                    info['info'] = file
                    pass
                elif 'metadata' in lower:
                    pass
                else:
                    info['maps'].append(file)
            # elif '.jpg' in file or '.jpeg' in file or '.png' in file:
            #     info['cover'] = file
        init_songs[song] = info
            
    return init_songs

def main(directory):
    path = './CustomLevels'
    songs = load_songs(path)
    for song in songs:
        print(songs[song])
        
        info = songs[song]
        # read the sound file
        data, samplerate = sf.read(info['egg'])
        #print(data, samplerate)
        # read the choreography 
        #maps = []
        #obstacles = []
        for map in info['maps']:
            with open(map, 'r') as f:
                line = f.readline()
                choreo = eval(line)
                #maps.append(choreo['_events'])
                #obstacles.append(choreo['_obstacles'])
                name = map.split('/')[-1]
                with open((directory + song + name + '.pkl').replace(' ', ''), 'wb') as f:
                    sample = {'name':song}
                    if 'info' in info:
                        with open(info['info'], 'rb') as f2:
                            lines = f2.readlines()
                            for line in lines:
                                entry = str(line).split('\"')
                                if '_beatsPerMinute' in entry:
                                    print('BPM', entry[2][2:-4])
                                    sample['BMP'] = entry[2][2:-4]
                                #print(line)
                                #chart_info = eval(line)
                                #sample['bpm'] = chart_info['_beatsPerMinute']
                        sample['songdata'] = np.asarray(data)
                        sample['samplerate'] = samplerate
                        sample['lights'] = choreo['_events']
                        sample['obstacles'] = choreo['_obstacles']
                        sample['notes'] = choreo['_notes']
                        pkl.dump(sample, f)

                #for key in choreo.keys():
                #    print(choreo[key])

def main2(directory):
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    #beat_samples = [join('./beat_samples', f) for f in listdir('./beat_samples') if isfile(join('./beat_samples', f))]
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
                    beatrate = 441
                    for entry in notes:
                        mapping = np.zeros([int(np.ceil(songlength*beatrate))])
                        time = entry['_time']/float(sample['BMP']) * 60 #static "60" BPM to match up to music
                        #songwindow = data[int((time-0.1)*samplerate):int(time*samplerate),:]
                        songwindow = data[int(time*samplerate)-int(samplerate/10):int(time*samplerate),:]
                        cutDirection = entry['_cutDirection'] #0-8
                        lineIndex = entry['_lineIndex'] #0-3
                        lineLayer = entry['_lineLayer'] #0-2
                        noteType = entry['_type'] #0-1 no bombs please -> remove 3 
                        cutDirection = entry['_cutDirection']
                        #print(int(np.round(time*beatrate)))
                        if noteType <= 1:
                            label = np.zeros([2,3,4,9])
                            print(noteType, lineLayer, lineIndex, cutDirection)
                            label[noteType, lineLayer, lineIndex, cutDirection] = 1
                            #mapping[int(np.round(time*beatrate))] = 1 
                            with open('./beat_samples/' + file.replace('.pkl','').replace('./samples/','') + str(time) + '.pkl', 'wb') as f2:
                                pkl.dump({'name':name, 'time':time, 'window':songwindow, 'label':label}, f2)
        except Exception as f:
            print(f)

if __name__ == "__main__":
    directory = './samples/'
    #main(directory)
    main2(directory) #second preprocessing needed 