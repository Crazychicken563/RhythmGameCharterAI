import os
import re
#Note: Currently only accepts fixed-BPM songs. This should be fixed at some point
#Currently banning songs with STOPS. This is definitely fixable
#Note: source_dir is currently hardcoded, probably a bad idea

source_dir = r"D:\Program Files\StepMania 5\Songs"
excludes = [
    "Konami official","Dance Dance Revolution 7th Mix -Max2-", #.dwi support needed
    "DDR X", "SPEIRMIX", "beta style", "gamma style", #Uses DDR X rating scale
    "Gull's Arrows - Infinity", #songs included in other Gull's Arrows packs
    "StepMania 5",#has weird/custom stuff, I should probably extract the originals
]
#ITG and DDR rating scales are close enough to ignore the difference
#Note that ITG 1/2 are the only official ones. 3, Rebirth, Rebirth+, Rebirth 2 are all fan-made

danceTypesFound = set()
songCount = 0
chartCount = 0
for (dirpath, dirnames, filenames) in os.walk(source_dir):
    name = os.path.relpath(dirpath, source_dir)
    excludeCheck = False
    for n in excludes:
        if n in name:
            excludeCheck = True
            break
    if excludeCheck:
        continue
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

    BPM = 0
    OFFSET = 0
    typeList = []
    skip = False
    #Many files don't use latin-1 encoding, but we don't care about that data
    with open(os.path.join(dirpath, chartname), encoding="latin-1") as chart:
        while True:
            line = chart.readline()
            if len(line) == 0:
                break
            if "#BPMS" in line:
                if "," in line:
                    skip = "Multiple BPMs detected"
                else:
                    bpmregex = re.search(r"=(\d+\.?\d*)", line)
                    BPM = bpmregex.group(1)#Note that negative BPMs can exist, but are weird
            if "#OFFSET" in line:
                offsetregex = re.search(r":(-?\d+\.?\d*)", line)
                OFFSET = offsetregex.group(1)
            if "#STOPS" in line:
                if "=" in line:
                    skip = "Stop detected"
            if "#NOTES" in line:
                danceType = chart.readline().strip()
                chart.readline() #author
                chart.readline() #difficulty name
                difficulty = chart.readline().strip()
                chart.readline() #groove radar
                position = chart.tell()
                
                danceTypesFound.add(danceType)
                if danceType == "dance-single:":
                    typeList.append((difficulty, position))
##            if "#METERTYPE" in line:
##                meterregex = re.search(":(.*);", line)
##                if meterregex.group(1) != "DDR":
##                    print(meterregex.group(1)+" meter from "+name)
    if skip:
        #print("-"+skip+" in "+name)
        continue
    if len(typeList) == 0:
        print("----No usable data found for "+name)
        continue
    songCount += 1
    for t in typeList:
        chartCount+=1
    
    newpath = os.path.join("ddr_data",name)
    os.makedirs(newpath, exist_ok=True)
    with open(os.path.join(newpath,"chart_data.dat"), mode='w', encoding="utf-8") as chart_data:
        chart_data.write("CHART="+os.path.join(dirpath, chartname)+"\n")
        chart_data.write("MUSIC="+os.path.join(dirpath, musicname)+"\n")
        chart_data.write("BPM="+BPM+"\n")
        chart_data.write("OFFSET="+OFFSET+"\n")
        for t in typeList:
            chart_data.write(t[0]+"@"+str(t[1])+"\n")
for t in danceTypesFound:
    print(t)
print(str(songCount)+","+str(chartCount))

#Errors in current set:
#no music: ITG3/Got to Have You
#dance-solo instead of dance-single? (Has full set of dance-singles though)
#    ITGR+/Tori no Uta (Noise Rave)
#no music: Jubo Impulsion (301-450)/KISS OF DEATH
#seems entirely missing? OD/(Sargon) Romantic summer

#fixes:
#Notice Me Benpai/La Ville electron, removed accent mark from e (was failing to load)
