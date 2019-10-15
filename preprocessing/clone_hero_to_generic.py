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
            currLine = notes.readline().strip()
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
                                mnd.write(f"{timestamp} {str(augment)} {timestamp} {endNote}\n")
                else:
                    if any(header in currLine for header in ["[ExpertSingle]", "[HardSingle]", "[MediumSingle]", "[EasySingle]"]):
                        print("Now scanning " + currLine)
                        notes.readline() #Skip the "{"
                        scanningHeader = True

                        this_difficulty_file = os.path.join("clone_hero_data", currLine + "_" + name + ".mnd")
                        print(this_difficulty_file)
                        success = False
                        mnd = open(this_difficulty_file, mode="w", encoding="utf-8")
                
                currLine = notes.readline().strip()

main()