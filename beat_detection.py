from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor
import os
import numpy as np
proc = TempoEstimationProcessor(fps=100)

source_dir = r"D:\Program Files\StepMania 5\Songs\Albumix 3.I"
np.set_printoptions(suppress=True)

for (dirpath, dirnames, filenames) in os.walk(source_dir):
    name = os.path.relpath(dirpath, source_dir)
    #print(name)
    musicname = ""
    chartname = ""
    if len(dirnames) > 5:
        print(">Subdirectories found in "+name+", continuing.")
        continue
    if dirnames:
        print("----<5 subdirectories in "+name+", investigate!")
    for f in filenames:
        if f.endswith(".sm") or f.endswith(".SM"):
            chartname = f#.ssc files are very similar, my packs don't need those .dwi files would need work
        elif f.endswith(".mp3") or f.endswith(".ogg"):
            musicname = f
    if musicname == "" or chartname == "":
        print ("----Music/Chart ("+musicname+","+chartname+") not found in "+name)
        continue
    print(musicname)
    act = RNNBeatProcessor()(os.path.join(dirpath, musicname))
    print(str(proc(act)))