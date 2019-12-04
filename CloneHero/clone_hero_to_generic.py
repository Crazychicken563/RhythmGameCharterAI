import os
import re
import pickle as pkl
import soundfile as sf
import numpy as np

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
                        with open("clone_hero_data/output/"+currSongName+".pkl", 'wb+') as f:
                            pkl.dump(currSong, f)
                        print("end of header for {}".format(currSongName))
                        currSong = None
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
                                currSong['n'].append({
                                    't': int(timestamp),
                                    'v': value,
                                    'd': int(duration)
                                })
                            #else:
                                #print("Unknown value note {}".format(datums))
                        elif datums[0] == "S":
                            # augments over 4 denote a unique type of note / note modifier
                            # augment 7 means that the previous note has star power.
                            # other augments currently unknown...
                            #print("star power for duration: {}".format(duration))
                            currSong['s'].append({
                                't': int(timestamp),
                                'v': 2,
                                'd': int(duration)
                            })
                else:
                    if any(header in currLine for header in ["[Song]"]):
                        n = notes.readline().strip()
                        while n != "}":
                            data = n.split("=")
                            print("{}".format(data))
                            if data[0].strip() == "MusicStream":
                                audioFile = data[1].strip().replace("\"", "")
                                print("AudioFile: {}".format(audioFile))

                            n = notes.readline().strip()
                    elif any(header in currLine for header in ["[ExpertSingle]", "[HardSingle]", "[MediumSingle]", "[EasySingle]"]):
                        print("Now scanning " + currLine)
                        notes.readline() #Skip the "{"
                        scanningHeader = True
                        mergedPathIntoName = name.replace("\\", "_")
                        currSongName = os.path.join(currLine + "_" + mergedPathIntoName)
                        print(currSongName)
                        print(audioFilePath)
                        data, samplerate = sf.read(audioFilePath)
                        print("sample rate: {}".format(samplerate))

                        currSong = {
                            'n': [],
                            's': [],
                            'sd': np.asarray(data),
                            'sr': samplerate
                        }

                currLine = notes.readline().strip()

main()