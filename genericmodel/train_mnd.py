import random
import sys
import io
import math
from generic_processing import *
import multiprocessing
from multiprocessing import Pool
#Note: Requires both ddr_finder and ddr_to_generic to be run first
#will dynamically construct both the MND's data and the output to compare against

source_dir = "../preprocessing/ddr_data"
songs_per = 256

#Returns a list of charts to be used later (this is cached in ddr_dataset.p)
def song_data(skipto = ""):
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        if skipto != "":
            if skipto in dirpath:
                skipto = ""
            else:
                continue
        name = os.path.relpath(dirpath, source_dir)
        print(name)
        if len(dirnames) > 5:
            print(">Subdirectories found in "+name+", continuing.")
            continue
        if dirnames:
            print("----<5 subdirectories in "+name+", investigate!")
        if not "chart_data.dat" in filenames:
            print("Chart data not found! "+name)
            continue
        with open(os.path.join(dirpath, "chart_data.dat"), encoding="utf-8") as chart_data:
            (a, chart_path) = chart_data.readline().strip().split(":",1)
            (b, music_path) = chart_data.readline().strip().split(":",1)
            (c, bpm)        = chart_data.readline().strip().strip(";").split(":")
            (d, offset)     = chart_data.readline().strip().split(":")
            assert (a, b, c, d) == ("CHART", "MUSIC", "BPM", "OFFSET")
            with open(chart_path, encoding="latin-1") as chart:
                while True:
                    tmp = chart_data.readline()
                    if len(tmp) < 3:
                        break
                    (difficulty, position) = tmp.strip().split("@")
                    difficulty = difficulty.strip(":")
                    this_difficulty_file = os.path.join(dirpath,"c"+str(difficulty)+"_"+position+".mnd")
                    if os.path.exists(this_difficulty_file):
                        yield (music_path,this_difficulty_file)


#Generator for all of the data involved in a single run of the network
def generate_song_inout_data(data_tuple):
    mnd_raw = []
    (song_path, data_file) = data_tuple
    with open(data_file, encoding="latin-1") as chart:
        ingored_song_path = chart.readline().strip()
        bpm_data = chart.readline().strip()
        offset = float(chart.readline())
        time_resolution = 48
        max_simultaneous = 1
        long_count = 0
        any_note_count = 0
        jump_count = 0
        current_holds = 0
        for line in chart:
            line_data = tuple(int(x) for x in line.split())
            t_res = beat_find(line_data[0])
            time_resolution = math.gcd(time_resolution,t_res)
            current_holds += line_data[2]-line_data[3]
            any_note_count += min(line_data[1]+line_data[2],1)
            long_count += line_data[2]
            if line_data[1]+line_data[2] >= 2:
                jump_count += 1
            max_simultaneous = max(max_simultaneous,line_data[1]+current_holds)
            assert(line_data[1]+line_data[2] <= 4)
            assert(current_holds >= 0)
            mnd_raw.append((line_data[0],line_data[1] + line_data[2],line_data[2],line_data[3]))
        assert(current_holds == 0)
    mnd_raw.append((0xFFFFFFFF,0,0,0))
    
    raw_audio = onset_strengths(data_file,song_path)
    #print(song_relpath(song_path))
    
    audio_length = (len(raw_audio)*SPECT_SKIP/SAMPLE_RATE)-(PADDING*2)
    note_freq = any_note_count/audio_length
    jump_freq = jump_count/any_note_count
    long_freq = long_count/any_note_count
    bpm_list = process_bpm(bpm_data)
    #print(note_freq,jump_freq,long_freq,data_file)
    
    anyhits = False

    bpm_id = 0
    bpm = bpm_list[bpm_id][1]
    next_bpm_time = bpm_list[bpm_id+1][0]*48
    
    now_sec = -offset
    now_beat = 0
    mnd_id = 0
    full_hist = [[max_simultaneous,note_freq/5,jump_freq,long_freq,time_resolution/48,
                bpm/400,0,0,0,1,
                0,0,0]]*NOTE_HISTORY#9 stats data, 3 output
  
    current_holds = 0
    last_found_beat = -192
    last_found_sec = -1
    
    while mnd_id < len(mnd_raw)-1 or now_sec < audio_length+4:
        #history = stats+output (+time apart) for previous 64 notes
        #const stats = (max concurrent, any note frequency, jump freq, long freq)
        #varying stats = (bpm, time apart sec, time apart beats, current holds, fractional beat)
        t_res = beat_find(now_beat)
        stats = (max_simultaneous,note_freq/5,jump_freq,long_freq,time_resolution/48,
                bpm/400,min(now_sec-last_found_sec,2)/2,min((now_beat-last_found_beat)/384,1),current_holds/4,t_res/48)
        hit = False
        if mnd_raw[mnd_id][0] == now_beat:
            hit = True
            anyhits = True
            out_dat = mnd_raw[mnd_id][1:]
            mnd_id += 1
            current_holds += out_dat[1] - out_dat[2]
            last_found_beat = now_beat
            last_found_sec = now_sec
            #print(now_sec,now_beat,extended_stats)
        else:
            if not now_beat < mnd_raw[mnd_id][0]:
                print("now:"+str(now_beat)+", goal:"+str(mnd_raw[mnd_id][0])+",bpmnext:"+str(next_bpm_time))
            assert(now_beat < mnd_raw[mnd_id][0])
            out_dat = (0,0,0)
        if anyhits and (hit or random.random() < 0.1*time_resolution):
            id_now = sec_to_id(now_sec)
            audio = raw_audio[id_now-AUDIO_BEFORE_LEN:id_now+AUDIO_AFTER_LEN]
            if len(audio) != AUDIO_AFTER_LEN+AUDIO_BEFORE_LEN:
                print(data_file)
                print("audio",now_sec,now_beat,id_now,len(raw_audio))
            #audio = audio.reshape((AUDIO_BEFORE_LEN+AUDIO_AFTER_LEN,1))
            #if (random.random() < 0.001):
            #    print (np.sum(raw_audio[id_now-4:id_now+1]), stats,out_dat)
            (out_note,out_long,out_long_end) = out_dat
            yield (audio,stats,out_note,out_long,out_long_end)
        
        minilast_beat = now_beat
        now_beat += time_resolution
        while now_beat > next_bpm_time:
            assert(minilast_beat <= next_bpm_time)
            now_sec += (next_bpm_time-minilast_beat)/(bpm/60*48)
            minilast_beat = next_bpm_time
            bpm_id += 1
            bpm = bpm_list[bpm_id][1]
            next_bpm_time = bpm_list[bpm_id+1][0]*48
        now_sec += (now_beat-minilast_beat)/(bpm/60*48)

#Take in a song_data tuple and return a full set of training data
def map_data_to_training_data(data_tuple):
    all_data = generate_song_inout_data(data_tuple)
    try:
        zipped_data = zip(*all_data)
        return zipped_data
    except ValueError:
        print("BAD DATA: ",song_relpath(data_tuple[0]))
        return ([],[],[],[],[],[])


def generate_dataset():
    gen = song_data()
    dataset = []
    try:
        while True:
            dataset.append(next(gen))
    except StopIteration:
        pass
    return dataset

def huge_full_dataset():
    #load dataset from disk (chart list)
    if os.path.exists("ddr_only_songs.p"):
        print("loading dataset")
        dataset = pickle.load(open("ddr_only_songs.p","rb"))
        print("dataset loaded!")
    else:
        dataset = generate_dataset()
        pickle.dump(dataset, open("ddr_only_songs.p","wb"))
    bag = dataset#.copy() : Don't need to copy when doing shuffle+slice rather than shuffle+pop
    song_count = len(dataset)
    max_iters = song_count//songs_per
    excess = song_count%songs_per
    #Creates "super-batches" of random songs due to memory limitations
    with Pool(processes=7) as pool:
        while True:
            print("reloading dataset bag: size="+str(song_count))
            random.shuffle(bag)
            next_dataSlice = bag[0:excess] #First set is smaller
            next_data_superbatch = pool.imap_unordered(map_data_to_training_data,next_dataSlice)
            for x in range(max_iters+1):
                #if not next_data_superbatch.ready():
                #    print(">>>DATA NOT READY!")
                #data_superbatch_complete = next_data_superbatch.get()
                audio_in_set   = []
                stats_set      = []
                out_note_set    = []
                out_long_set    = []
                out_long_end_set= []
                for data in next_data_superbatch:
                    dat = tuple(data)
                    if len(dat) != 5:
                        print("skip")
                        continue
                    (audio_in, stats, out_note,out_long,out_long_end) = dat
                    audio_in_set.extend(audio_in)
                    stats_set.extend(stats)
                    out_note_set.extend(out_note)
                    out_long_set.extend(out_long)
                    out_long_end_set.extend(out_long_end)
                    
                print("Data prepared (",x*songs_per+excess,"/",song_count,")")
                audio_in_set=np.array(audio_in_set)
                stats_set       =np.array(stats_set       )
                out_note_set    =np.array(out_note_set    )
                out_long_set    =np.array(out_long_set    )
                out_long_end_set=np.array(out_long_end_set)
                print("Data sent")
                if x < max_iters:
                    next_dataSlice = bag[x*songs_per+excess:(x+1)*songs_per+excess]
                    next_data_superbatch = pool.imap_unordered(map_data_to_training_data,next_dataSlice)
                yield ((audio_in_set,stats_set),
                    (out_note_set,out_long_set,out_long_end_set))


if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import LambdaCallback
    from tensorflow.keras import layers
    from tensorflow.keras import models
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.models import load_model
    model_make = True
    if len(sys.argv) > 1:
        if sys.argv[1] == "LOAD":
            print("Loading model")
            model = load_model("song_model.h5")
            #If loading from disk, maybe assume the model is reasonably-trained and use the slower but "better" SGD algorithm?
            optimizer = keras.optimizers.Nadam()
            if len(sys.argv) > 2:
                if sys.argv[2] == "SGD":
                    optimizer = keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True)
                if sys.argv[2] == "RMS":
                    optimizer = RMSprop(learning_rate=0.0002)
                if sys.argv[2] == "KEEP":
                    optimizer = None
            if optimizer is not None:
                model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
            model_make = False
    if model_make:
        def doublePool(x, pool_size=2,strides=None):
            return layers.concatenate([layers.MaxPooling1D(pool_size,strides)(x),layers.AveragePooling1D(pool_size,strides)(x)])
        
        audio_in = layers.Input(shape=(AUDIO_BEFORE_LEN+AUDIO_AFTER_LEN,7,),name="audio_in")
        audio = doublePool(audio_in, pool_size=2,strides=1)
        audio = layers.Conv1D(160,4, activation='elu')(audio)
        audio = doublePool(audio)
        audio = layers.Conv1D(128,8, activation='elu')(audio)
        audio = doublePool(audio)
        audioa = layers.LocallyConnected1D(12,6,strides=1, activation='elu')(audio)
        audioa = layers.LocallyConnected1D(16,18,strides=1, activation='elu')(audioa)
        audioa = layers.Flatten()(audioa)
        def crop_fn(x):
            l = x.shape[1]
            return x[:,l//2:]
        audiob = layers.Lambda(crop_fn)(audio)
        audiob = layers.LocallyConnected1D(32,4,strides=1, activation='elu')(audiob)
        audiob = layers.LocallyConnected1D(48,12,strides=1, activation='elu')(audiob)
        audiob = layers.Flatten()(audiob)
        audio = layers.concatenate([audioa,audiob])
        audio = layers.Dense(512, activation='elu')(audio)
         
        stats_input = layers.Input(shape=(10,),name="stats_input")
        x = layers.Dense(32, activation='elu')(stats_input)
        x = layers.concatenate([x,audio])
        x = layers.Dense(768, activation='elu')(x)
        x = layers.Dense(512, activation='elu')(x)
        x = layers.Dense(384, activation='elu')(x)
        x = layers.Dense(256, activation='elu')(x)
        x = layers.Dense(196, activation='elu')(x)
        x = layers.Dense(128, activation='elu')(x)
        x = layers.Dense(128, activation='elu')(x)
        out_basic = layers.Dense(5, activation='softmax', name = "out_basic")(x) #Basic Notes, Start Long, End long
        out_long = layers.Dense(5, activation='softmax', name = "out_long")(x) #Basic Notes, Start Long, End long
        out_long_end = layers.Dense(5, activation='softmax', name = "out_long_end")(x) #Basic Notes, Start Long, End long

        optimizer = keras.optimizers.Nadam()
        model = models.Model(inputs=[audio_in,stats_input],outputs=[out_basic,out_long,out_long_end])
        #audio_before/after = 1/2 and 1/8 sec
        #history = stats+output (+time apart) for previous 64 notes
        #const stats = (max concurrent, any note frequency, jump freq, long freq)
        #varying stats = (bpm, time apart sec, time apart beats, current holds, fractional beat)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    model.summary()
    huge_gen = huge_full_dataset()
    while True: #huge_full_dataset will keep repeating after going through all songs
        (ins, outs) = next(huge_gen)
        model.fit(ins,outs,epochs=1,batch_size=256)
        del ins
        del outs
        print("Saving model")
        if os.path.exists("song_modelBACKUP.h5"):
            os.remove("song_modelBACKUP.h5")
        if os.path.exists("song_model.h5"):
            os.rename("song_model.h5","song_modelBACKUP.h5")
        model.save("song_model.h5")
        print("Save complete")