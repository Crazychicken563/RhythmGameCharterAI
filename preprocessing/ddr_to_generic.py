import os
import re
#Note: Requires ddr_finder to be run first
#mnd (Music Note Data) format:

#Header is 3 lines
#1) Filepath to music file (string)
#2) BPM (float)
#3) Offset from music start(float)

#Main data:
#A time point is 1/192 of a measure (1/48 of a beat)
#Every line is an integer representing the number of time points after the start*
#  *start = music start plus offset (in chart_data.dat)
#Then three ints which must be 0 or 1 representing (individual note, start of held note, end of held note)
# lines of "000" are always omitted
#each value separated by a space

#DDC used a constant 125 BPM and 1/48 measure resolution
source_dir = "ddr_data"
def main():
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        name = os.path.relpath(dirpath, source_dir)
        if len(dirnames) > 5:
            print(">Subdirectories found in "+name+", continuing.")
            continue
        if dirnames:
            print("----<5 subdirectories in "+name+", investigate!")
        if not "chart_data.dat" in filenames:
            print("Chart data not found! "+name)
            continue
        with open(os.path.join(dirpath, "chart_data.dat"), encoding="utf-8") as chart_data:
            (a, chart_path) = chart_data.readline().strip().split("=")
            (b, music_path) = chart_data.readline().strip().split("=")
            (c, bpm)        = chart_data.readline().strip().split("=")
            (d, offset)     = chart_data.readline().strip().split("=")
            assert (a, b, c, d) == ("CHART", "MUSIC", "BPM", "OFFSET")
            with open(chart_path, encoding="latin-1") as chart:
                while True:
                    tmp = chart_data.readline()
                    if len(tmp) < 3:
                        break
                    (difficulty, position) = tmp.strip().split("@")
                    difficulty = difficulty.strip(":")
                    chart.seek(int(position))
                    this_difficulty_file = os.path.join(dirpath,"c"+str(difficulty)+".mnd")
                    #print(this_difficulty_file)
                    with open(this_difficulty_file, mode="w", encoding="utf-8") as mnd:
                        mnd.write(music_path+"\n")
                        mnd.write(bpm+"\n")
                        mnd.write(offset+"\n")
                        write_mnd_data(chart, mnd)


def bl(value):#returns 0 or 1 based on true/false input plus a prefixed space
    return " 1" if value else " 0"

def write_mnd_data(chart, mnd):
    #assumes chart has already been seek()ed to the right position (first line of note data)
    #assumes mnd has had header lines written
    time_point = 0
    stored_lines = []
    while True:
        line = chart.readline().strip()
        if ";" in line or "," in line:
            #process a measure
            count = len(stored_lines)
            if not count in [4, 8, 12, 16, 24, 32, 48, 64, 96, 192]:
                print("bad count("+str(count)+") at "+str(chart.tell()))
                break
            time_resolution = 192/count
            for notes in stored_lines:
                note = "1" in notes or "M" in notes
                #Counting mines (M) as notes.
                start_long = "2" in notes or "4" in notes
                #Counting rolls (4) as long notes- probably not a good idea
                end_long = "3" in notes
                if (note or start_long or end_long):
                    mnd.write(str(time_point)+bl(note)+bl(start_long)+bl(end_long)+"\n")
                time_point += time_resolution
            stored_lines = []
        else:
            if len(line) == 4:
                stored_lines.append(line)
        if ";" in line:
            break

main()