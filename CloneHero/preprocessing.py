import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import cv2
import soundfile as sf
import csv
import pickle as pkl

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
                    label = np.zeros([beatrate+1])
                    for entry in notes:
                        if entry['_time'] >= time and entry['_time'] < time + 1 and entry['_type'] < 3:
                            # bandied fix for the weird omnidirectional boxes (8) - don't want a massive tensor, could set to index 4
                            if (entry['_cutDirection']) > 3:
                                entry['_cutDirection'] = 4
                            #print(np.floor((entry['_time'] % 1) * beatrate).astype(int), ((entry['_time'] % 1) * beatrate))
                            label[np.floor((entry['_time'] % 1) * beatrate).astype(int)] = 1
                            with open('./samples_window/' + file.replace('.pkl','').replace('./samples/','') + str(time) + '.pkl', 'wb') as f2:
                                pkl.dump({'name':name, 'time':entry['_time'], 'window':window, 'label':label}, f2)

if __name__ == "__main__":
    directory = './samples/'
    #main(directory)
    main2(directory) #second preprocessing needed 