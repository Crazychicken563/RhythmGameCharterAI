import os
import re

source_dir = "../clone_hero_data/clonehero-win64/songs"
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
            currLine = notes.readline()
            
            (a, chart_path) = notes.readline().strip().split("=")
            (b, music_path) = notes.readline().strip().split("=")
            (c, bpm)        = notes.readline().strip().split("=")
            (d, offset)     = notes.readline().strip().split("=")
            assert (a, b, c, d) == ("CHART", "MUSIC", "BPM", "OFFSET")
            with open(chart_path, encoding="latin-1") as chart:
                while True:
                    tmp = chart_data.readline()
                    if len(tmp) < 3:
                        break
                    (difficulty, position) = tmp.strip().split("@")
                    difficulty = difficulty.strip(":")
                    chart.seek(int(position))
                    this_difficulty_file = os.path.join(dirpath,"c"+str(difficulty)+"_"+position+".mnd")
                    #print(this_difficulty_file)
                    success = False
                    with open(this_difficulty_file, mode="w", encoding="utf-8") as mnd:
                        mnd.write(music_path+"\n")
                        mnd.write(bpm+"\n")
                        mnd.write(offset+"\n")
                    if not success:
                        os.remove(this_difficulty_file)

main()