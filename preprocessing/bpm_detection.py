# Copyright 2012 Free Software Foundation, Inc.
#
# This file is part of The BPM Detector Python
#
# The BPM Detector Python is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# The BPM Detector Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with The BPM Detector Python; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

import wave, array, math, time, argparse, sys
import numpy, pywt
from scipy import signal
import pdb
import matplotlib.pyplot as plt

def read_wav(filename):

    #open file, get metadata for audio
    try:
        wf = wave.open(filename,'rb')
    except IOError as e:
        print(e)
        return

    # typ = choose_type( wf.getsampwidth() ) #TODO: implement choose_type
    nsamps = wf.getnframes();
    assert(nsamps > 0);

    fs = wf.getframerate()
    assert(fs > 0)

    # read entire file and make into an array
    samps = list(array.array('i',wf.readframes(nsamps)))
    #print 'Read', nsamps,'samples from', filename
    try:
        assert(nsamps == len(samps))
    except AssertionError as e:
        print(nsamps, "not equal to", len(samps))
    
    return samps, fs
    
# print an error when no data can be found
def no_audio_data():
    print("No audio data for sample, skipping...")
    return None, None
    
# simple peak detection
def peak_detect(data):
    max_val = numpy.amax(abs(data)) 
    peak_ndx = numpy.where(data==max_val)
    if len(peak_ndx[0]) == 0: #if nothing found then the max must be negative
        peak_ndx = numpy.where(data==-max_val)
    return peak_ndx
    
def bpm_detector(data,fs):
    cA = [] 
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2**(levels-1);
    min_ndx = 60./ 220 * (fs/max_decimation)
    max_ndx = 60./ 40 * (fs/max_decimation)
    
    for loop in range(0,levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA,cD] = pywt.dwt(data,'db4');
            cD_minlen = len(cD)//max_decimation+1;
            cD_sum = numpy.zeros(cD_minlen);
        else:
            [cA,cD] = pywt.dwt(cA,'db4');
        # 2) Filter
        cD = signal.lfilter([0.01],[1 -0.99],cD);

        # 4) Subtractargs.filename out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[::(2**(levels-loop-1))]);
        cD = cD - numpy.mean(cD);
        # 6) Recombine the signal before ACF
        #    essentially, each level I concatenate 
        #    the detail coefs (i.e. the HPF values)
        #    to the beginning of the array
        cD_sum = cD[0:cD_minlen] + cD_sum;

    if [b for b in cA if b != 0.0] == []:
        return no_audio_data()
    # adding in the approximate data as well...    
    cA = signal.lfilter([0.01],[1 -0.99],cA);
    cA = abs(cA);
    cA = cA - numpy.mean(cA);
    cD_sum = cA[0:cD_minlen] + cD_sum;
    
    # ACF
    correl = numpy.correlate(cD_sum,cD_sum,'full') 
    
    midpoint = len(correl) // 2
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[int(min_ndx):int(max_ndx)]);
    if len(peak_ndx) > 1:
        return no_audio_data()
        
    peak_ndx_adjusted = peak_ndx[0]+min_ndx;
    bpm = 60./ peak_ndx_adjusted * (fs/max_decimation)
    print(bpm)
    return bpm,correl

def get_bpms(samps, fs, window):
    data = []
    correl=[]
    bpm = 0
    n = 0;
    nsamps = len(samps)
    window_samps = int(window*fs)
    samps_ndx = 0;  #first sample in window_ndx 
    max_window_ndx = nsamps // window_samps;
    print("Max window index: " + str(max_window_ndx))
    bpms = numpy.zeros(max_window_ndx)

    #iterate through all windows
    for window_ndx in range(0,max_window_ndx):

        #get a new set of samples
        #print n,":",len(bpms),":",max_window_ndx,":",fs,":",nsamps,":",samps_ndx
        data = samps[samps_ndx:samps_ndx+window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError( str(len(data) ) ) 
        
        bpm, correl_temp = bpm_detector(data,fs)
        if bpm == None:
            continue
        bpms[window_ndx] = bpm
        correl = correl_temp
        
        #iterate at the end of the loop
        samps_ndx = samps_ndx+window_samps;
        n=n+1; #counter for debug...

    return bpms
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process .wav file to determine the Beats Per Minute.')
    parser.add_argument('--filename', required=True,
                   help='.wav file for processing')

    args = parser.parse_args()
    samps,fs = read_wav(args.filename)
    print(len(samps))
    print(fs)
    bpmsForTime = {}

    maxWindowSize = 30
    step = 5
    stop = maxWindowSize + step

    endOfSong = len(samps) // fs

    for i in range(0, endOfSong, step):
        bpmsForTime[i] = []

    print(bpmsForTime)

    # start at step rather than 0 because we dont want to calculate window size of 0
    for window in range(step, stop, step):
        print("Get BPMs for window size " + str(window))
        bpms = get_bpms(samps, fs, window)
        print(bpms)
        for i in range(0, len(bpms)):
            bpm = bpms[i]
            startTime = i * window
            for time in range(startTime, startTime + window, step):
                print("BPM @ time " + str(time) + " = " + str(bpm))
                bpmsForTime[time].append(bpm)

    print(bpmsForTime)
        
    differentThreshold = 3
    stdThreshold = 15

    test = {0: [195.5470156674102, 218.54632827751917, 198.3616389063054, 198.83863915835497, 198.0053879017116, 194.97065995016214], 5: [213.8826019223421, 218.54632827751917, 198.3616389063054, 198.83863915835497, 198.0053879017116, 194.97065995016214], 10: [193.03621169916434, 195.20079405531564, 198.3616389063054, 198.83863915835497, 198.0053879017116, 194.97065995016214], 15: [194.97065995016214, 195.20079405531564, 49.02608121492242, 198.83863915835497, 198.0053879017116, 194.97065995016214], 20: [179.91099023365064, 196.3596621421054, 49.02608121492242, 97.72754744349088, 198.0053879017116, 194.97065995016214], 25: [57.37795405978694, 196.3596621421054, 49.02608121492242, 97.72754744349088, 65.28292407209698, 194.97065995016214], 30: [97.46835443037975, 65.27004117219666, 65.27004117219666, 97.72754744349088, 65.28292407209698, 49.00428993783967], 35: [193.37479071992348, 65.27004117219666, 65.27004117219666, 97.72754744349088, 65.28292407209698, 49.00428993783967], 40: [194.28350198915976, 194.51201582506886, 65.27004117219666, 130.65845468747196, 65.28292407209698, 49.00428993783967], 45: [201.381009049899, 194.51201582506886, 129.8378031154649, 130.65845468747196, 65.28292407209698, 49.00428993783967], 50: [129.9908891151723, 129.78685454383304, 129.8378031154649, 130.65845468747196, 129.9908891151723, 49.00428993783967], 55: [196.3596621421054, 129.78685454383304, 129.8378031154649, 130.65845468747196, 129.9908891151723, 49.00428993783967], 60: [130.65845468747196, 48.98977317866304, 49.011551544134974, 49.011551544134974, 129.9908891151723, 49.011551544134974], 65: [129.53271028037383, 48.98977317866304, 49.011551544134974, 49.011551544134974, 129.9908891151723, 49.011551544134974], 70: [129.68507726033258, 130.65845468747196, 49.011551544134974, 49.011551544134974, 129.9908891151723, 49.011551544134974], 75: [130.7100899962277, 130.65845468747196, 130.45232076587962, 49.011551544134974, 49.09157148350796, 49.011551544134974], 80: [130.29814665592264, 130.04199803413456, 130.45232076587962, 49.011551544134974, 49.09157148350796, 49.011551544134974], 85: [98.04621707202047, 130.04199803413456, 130.45232076587962, 49.011551544134974, 49.09157148350796, 49.011551544134974], 90: [98.162612813145, 78.36918006655968, 49.011551544134974, 49.011551544134974, 49.09157148350796, 49.011551544134974], 95: [78.38775356307971, 78.36918006655968, 49.011551544134974, 49.011551544134974,
49.09157148350796, 49.011551544134974], 100: [194.1694462975317, 194.1694462975317, 49.011551544134974, 129.482000818549, 129.58345947678663, 49.011551544134974], 105: [202.24297506879017, 194.1694462975317, 129.53271028037383, 129.482000818549, 129.58345947678663, 49.011551544134974], 110: [128.47608453837597, 128.47608453837597, 129.53271028037383, 129.482000818549, 129.58345947678663, 49.011551544134974], 115: [49.12073446518379, 128.47608453837597, 129.53271028037383, 129.482000818549, 129.58345947678663, 49.011551544134974], 120: [130.55530636045575, 48.96076544721738, 48.96076544721738, 49.06244310941198, 129.58345947678663, 49.06244310941198], 125: [197.65041423332883, 48.96076544721738, 48.96076544721738, 49.06244310941198, 49.06244310941198, 49.06244310941198], 130: [144.97619095056882, 49.13532895314368, 48.96076544721738, 49.06244310941198, 49.06244310941198, 49.06244310941198], 135: [79.3468186031296, 49.13532895314368, 49.06244310941198, 49.06244310941198, 49.06244310941198, 49.06244310941198], 140: [49.12073446518379, 48.95351888106242, 49.06244310941198, 48.946274459683984, 49.06244310941198, 49.06244310941198], 145: [48.98251802387027, 48.95351888106242, 49.06244310941198, 48.946274459683984, 49.06244310941198, 49.06244310941198], 150: [192.36259814418275, 197.41447136384602, 196.71001054310508, 48.946274459683984, 48.98977317866304, 48.98977317866304], 155: [195.20079405531564, 197.41447136384602, 196.71001054310508, 48.946274459683984, 48.98977317866304, 48.98977317866304], 160: [196.24315650368135, 48.96076544721738, 196.71001054310508, 48.98251802387027, 48.98977317866304, 48.98977317866304], 165: [97.95910125065629, 48.96076544721738, 49.00428993783967, 48.98251802387027, 48.98977317866304, 48.98977317866304], 170: [195.77851319719105, 49.011551544134974, 49.00428993783967, 48.98251802387027, 48.98977317866304, 48.98977317866304], 175: [65.65875314690993, 49.011551544134974, 49.00428993783967, 48.98251802387027, 130.04199803413456, 48.98977317866304], 180: [129.9398203539349, 129.9398203539349, 131.48954625128752, 130.55530636045575, 130.04199803413456, 48.8739479994358], 185: [129.88879170311134, 129.9398203539349, 131.48954625128752, 130.55530636045575, 130.04199803413456, 48.8739479994358],
190: [198.3616389063054, 195.77851319719105, 131.48954625128752, 130.55530636045575, 130.04199803413456, 48.8739479994358], 195: [195.77851319719105, 195.77851319719105, 193.9417361869986, 130.55530636045575, 130.04199803413456, 48.8739479994358], 200: [49.09157148350796, 49.09157148350796, 193.9417361869986, 193.9417361869986, 48.98977317866304, 48.8739479994358], 205: [195.66269595848237, 49.09157148350796, 193.9417361869986, 193.9417361869986, 48.98977317866304, 48.8739479994358], 210: [57.37795405978694, 194.05552444195538, 194.1694462975317, 193.9417361869986, 48.98977317866304, 48.946274459683984], 215: [194.05552444195538, 194.05552444195538, 194.1694462975317, 193.9417361869986, 48.98977317866304, 48.946274459683984], 220: [98.6898319567075, 49.02608121492242, 194.1694462975317, 48.946274459683984, 48.98977317866304, 48.946274459683984], 225: [199.19789756084208, 49.02608121492242, 199.3179390253924, 48.946274459683984, 48.946274459683984, 48.946274459683984], 230: [200.40485829959516, 200.40485829959516, 199.3179390253924, 48.946274459683984, 48.946274459683984, 48.946274459683984], 235: [200.0412371134021, 200.40485829959516, 199.3179390253924, 48.946274459683984, 48.946274459683984, 48.946274459683984], 240: [48.95351888106242, 49.12803062526584, 48.95351888106242, 201.01384016133045, 48.946274459683984], 245: [196.01055949142042, 49.12803062526584, 48.95351888106242, 201.01384016133045, 48.946274459683984], 250: [78.61132417920767, 200.8917478810635, 48.95351888106242, 201.01384016133045], 255: [217.97022436569515, 200.8917478810635, 201.01384016133045]}

    for time in bpmsForTime:
        bpms = bpmsForTime[time]
        bpms.sort()
        print("Calc bpms for time " + str(time))
        # Now we have a range of possible bpms for a given time.
        # We need to check if they are multiples of each other
        # to avoid discarding valid bpm values
        maxBPM = max(bpms)
        currMin = bpms[0]
        bestSTD = stdThreshold
        bestBPMs = None
        while currMin < maxBPM:
            bpmSTD = numpy.std(bpms)
            if (bpmSTD < bestSTD):
                bestSTD = bpmSTD
                bestBPMs = bpms.copy()

            #print("Standard deviation of bpms: " + str(bpmSTD))

            for i in range(0, len(bpms)):
                if (bpms[i] - currMin > differentThreshold):
                    continue
                bpms[i] *= 2

            bpms.sort()
            currMin = bpms[0]

        print("Best STD: " + str(bestSTD))
        print("Best BPMS: " + str(bestBPMs))

        if bestBPMs == None or len(bestBPMs) == 0:
            bpmsForTime[time] = 0
            continue

        avgBPM = numpy.average(bestBPMs)
        medianBPM = numpy.median(bestBPMs)
        difference = medianBPM - avgBPM
        while abs(difference) > differentThreshold:
            if (difference > 0):
                bestBPMs.pop(0)
            else:
                bestBPMs.pop(len(bestBPMs) - 1)

            print(bestBPMs)

            avgBPM = numpy.average(bestBPMs)
            medianBPM = numpy.median(bestBPMs)
            difference = medianBPM - avgBPM
        
        bestBPM = numpy.median(bestBPMs)
        bpmsForTime[time] = bestBPM

    print(bpmsForTime)
