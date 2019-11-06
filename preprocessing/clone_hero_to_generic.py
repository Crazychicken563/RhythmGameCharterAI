import os
import re

source_dir = "clone_hero_data/clonehero-win64/songs"
def main():
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        name = os.path.relpath(dirpath, source_dir)
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
        with open(os.path.join(dirpath, "notes.chart"), encoding="utf-8") as notes:
            scanningHeader = False
            try:
                currLine = notes.readline().strip()
            except UnicodeDecodeError as e:
                print(e)
                continue
            while currLine:
                if scanningHeader:
                    if currLine == "}":
                        scanningHeader = False
                        print("end of header")
                    else:
                        (timestamp, data) = currLine.split("=")
                        timestamp = timestamp.strip()
                        datums = data.strip().split(" ")
                        if datums[0] == "N":
                            #These are the only things we care about for now
                            augment = int(datums[1].strip())
                            duration = datums[2].strip()
                            if augment <= 4:
                                endNote = str(int(timestamp) + int(duration))
                                # mnd will always be defined by this point since scanningHeader
                                # can never be true without mnd being instantiated
                                mnd.write(f"{timestamp} {str(augment)} {timestamp} {endNote}\n")
                            else:
                                # augments over 4 denote a unique type of note / note modifier
                                # augment 7 means that the previous note has star power.
                                # other augments currently unknown...
                                if augment == 7:
                                    # How should we write a star power not modifier to our generic data?
                                    print("star power for duration: " + duration)
                else:
                    if any(header in currLine for header in ["[ExpertSingle]", "[HardSingle]", "[MediumSingle]", "[EasySingle]"]):
                        print("Now scanning " + currLine)
                        notes.readline() #Skip the "{"
                        scanningHeader = True
                        mergedPathIntoName = name.replace("\\", "_")
                        this_difficulty_file = os.path.join("clone_hero_data\\output", currLine + "_" + mergedPathIntoName + ".mnd")
                        print(this_difficulty_file)
                        success = False
                        mnd = open(this_difficulty_file, mode="w+", encoding="utf-8")

                currLine = notes.readline().strip()

main()