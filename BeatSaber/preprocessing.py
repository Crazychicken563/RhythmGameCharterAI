import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import soundfile as sf
import csv

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
                    #info['info'] = file
                    pass
                elif 'metadata' in lower:
                    pass
                else:
                    info['maps'].append(file)
            # elif '.jpg' in file or '.jpeg' in file or '.png' in file:
            #     info['cover'] = file
        init_songs[song] = info
            
    return init_songs

if __name__ == "__main__":
    path = './CustomLevels'
    songs = load_songs(path)
    with open('songs.csv', 'w') as csvfile:
        fieldnames = ['song', 'info', 'maps']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for song in songs:
            print(songs[song])
            info = songs[song]
            # read the sound file
            data, samplerate = sf.read(info['egg'])
            #print(data, samplerate)
            # read the choreography 
            maps = []
            for map in info['maps']:
                with open(map, 'r') as f:
                    line = f.readline()
                    choreo = eval(line)
                    maps.append(choreo)
                    #for key in choreo.keys():
                    #    print(choreo[key])
            csvwriter.writerow({'song':song, 'info':info, 'maps':maps})
