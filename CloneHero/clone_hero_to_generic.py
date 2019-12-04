import os
import re
import pickle as pkl
import soundfile as sf
import numpy as np

def safeAdd(src, key, val):
    if key in src:
        src[key].update(val)
    else:
        src[key] = val

source_dir = "clone_hero_data/clonehero-win64/songs"
def main():
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        name = os.path.relpath(dirpath, source_dir)
        audioFilePath = None
        if not filenames:
            continue
        if "notes.mid" in filenames:
            print("we aren't parsing midi files right now")
            continue
        if not "notes.chart" in filenames:
            print("Chart data not found! " + name)
            print(filenames)
            continue
        else:
            print("Parsing " + name)
            foundOGG = False
            for filename in filenames:
                if (filename.endswith(".ogg")):
                    foundOGG = True
                    audioFilePath = os.path.abspath(source_dir + "\\" + name + "\\" + os.path.join(filename))
            if foundOGG == False:
                print("NO AUDIO FILE FOUND")
                continue
        with open(os.path.join(dirpath, "notes.chart"), encoding="utf-8") as notes:
            scanningHeader = False
            currSong = None
            currSongName = None
            try:
                currLine = notes.readline().strip()
            except UnicodeDecodeError as e:
                print(e)
                continue
            while currLine:
                if scanningHeader:
                    if currLine == "}":
                        scanningHeader = False
                        samplerate = currSong['sr']
                        songlength = currSong['sd'].shape[0]/samplerate
                        # yeah not dealing with 48000 right now
                        if samplerate == 44100:
                            os.mkdir("clone_hero_data/output/"+currSongName)
                            timestamps = list(currSong['ts'].keys())
                            for i in range(0, len(timestamps)) : 
                                timestamps[i] = int(timestamps[i])
                            timestamps.sort() 
                            print(name, samplerate)
                            beatrate = 441
                            mapping = np.zeros(int(np.ceil(songlength*beatrate)))
                            currBPM = 0
                            for timestamp in timestamps:
                                data = currSong['ts'][str(timestamp)]
                                #print("{}".format(data))
                                if "B" in data:
                                    currBPM = data["B"]
                                    print("currBPM {}".format(currBPM))
                                
                                time = float(timestamp)/float(currBPM) * 60 #static "60" BPM to match up to music
                                if "N" in data:
                                    mapping[int(np.round(time*beatrate)), int(data["N"]["v"])] = 1 
                                #print(int(np.round(time*beatrate)))
                            for time in range(int(np.floor(songlength))):
                                songwindow = currSong['sd'][time*samplerate:(time+1)*samplerate]
                                mapwindow = mapping[time*beatrate:(time+1)*beatrate]
                                
                                with open("clone_hero_data/output/"+currSongName+"/"+str(time)+".pkl", 'wb+') as f:
                                    pkl.dump({'name':name, 'time':time, 'window':songwindow, 'label':mapwindow}, f)
    
                        for timestamp in currSong['ts']:
                            currSong['ts'][timestamp].pop("N", None)
                            currSong['ts'][timestamp].pop("S", None)

                        for timestamp in list(currSong['ts'].keys()):
                            if len(currSong['ts'][timestamp].keys()) == 0:
                                currSong['ts'].pop(str(timestamp))

                        print("end of header for {}".format(currSongName))
                    else:
                        (timestamp, data) = currLine.split("=")
                        timestamp = timestamp.strip()
                        datums = data.strip().split(" ")
                        if datums[0] == "N":
                            #These are the only things we care about for now
                            value = int(datums[1].strip())
                            duration = datums[2].strip()
                            if value <= 4:
                                # mnd will always be defined by this point since scanningHeader
                                # can never be true without mnd being instantiated
                                safeAdd(currSong['ts'], str(timestamp), {
                                    "N": {
                                        'v': value,
                                        'd': int(duration)
                                    }
                                })
                            #else:
                                #print("Unknown value note {}".format(datums))
                        elif datums[0] == "S":
                            # augments over 4 denote a unique type of note / note modifier
                            # augment 7 means that the previous note has star power.
                            # other augments currently unknown...
                            #print("star power for duration: {}".format(duration))
                            safeAdd(currSong['ts'], str(timestamp), {
                                "S": {
                                    'v': 2,
                                    'd': int(duration)
                                }
                            })
                else:
                    #if any(header in currLine for header in ["[Song]"]):
                    #    print("Found Song header")
                    if any(header in currLine for header in ["[SyncTrack]"]):
                        notes.readline() #Skip the "{"

                        print(audioFilePath)
                        songdata, samplerate = sf.read(audioFilePath)
                        print("sample rate: {}".format(samplerate))
                        currSong = {
                            'ts': {},
                            'sd': np.asarray(songdata),
                            'sr': samplerate
                        }

                        currLine = notes.readline().strip()
                        while currLine != "}":
                            (timestamp, data) = currLine.split("=")
                            timestamp = timestamp.strip()
                            datums = data.strip().split(" ")
                            if datums[0] == "B":
                                #print("{}".format(datums))
                                #print(currSong)
                                safeAdd(currSong['ts'], str(timestamp), {
                                    "B": int(datums[1].strip())
                                })

                            currLine = notes.readline().strip()
                    elif any(header in currLine for header in ["[ExpertSingle]", "[HardSingle]", "[MediumSingle]", "[EasySingle]"]):
                        print("Now scanning " + currLine)
                        notes.readline() #Skip the "{"
                        scanningHeader = True
                        mergedPathIntoName = name.replace("\\", "_")
                        currSongName = os.path.join(currLine + "_" + mergedPathIntoName)
                        print(currSongName)

                currLine = notes.readline().strip()

main()