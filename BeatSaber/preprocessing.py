import numpy as np
from os import listdir
from os.path import isfile, join, isdir

def load_songs(path):
    songs = [f for f in listdir(path) if isdir(join(path, f))]

    print('Songs:', songs)

    init_songs = {song:None for song in songs}

    for song in songs:
        files = [join(path,song,f) for f in listdir(join(path,song)) if isfile(join(path,song,f))]
        print(song, files)
        attributes = ['info', 'egg', 'maps', 'cover']
        info = {attribute:None for attribute in attributes}
        info['maps'] = []
        for file in files:
            if '.egg' in file:
                info['egg'] = file
            elif '.dat' in file:
                if 'info' in file:
                    info['info'] = file
                else:
                    info['maps'].append(file)
            elif 'cover' in file:
                info['cover'] = file
        init_songs[song] = info
            
    return init_songs

if __name__ == "__main__":
    path = './CustomLevels'
    songs = load_songs(path)
    print(songs)
