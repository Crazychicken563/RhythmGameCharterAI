from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import pickle as pkl
import os
import os.path
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from trains import Task
from os import listdir
from os.path import isfile, join

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True

class audio(Dataset):
    def __init__(self, directory):
        self.items = []

        files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
        type_max = 0
        value_max = 0
        for file in files:
            with open(file, 'rb') as f:
                ## NOTE For reconstruction, this data needs to be added to *items*
                sample = pkl.load(f)
                #name = sample['name'] # this is needed for reconstruction of the song
                data = sample['songdata']
                samplerate = sample['samplerate']
                songlength = data.shape[0]/samplerate
                notes = sample['notes']
                #lights = sample['lights']
                #obstacles = sample['obstacles'] # to be added in the future
                for time in range(int(songlength)):
                    window = data[samplerate*time:samplerate*(time+1)]
                    label = np.zeros([window.shape[0], 5, 3, 2, 5])
                    for entry in notes:
                        if entry['_time'] >= time and entry['_time'] < time + 1 and entry['_type'] < 3:
                            # bandied fix for the weird omnidirectional boxes (8) - don't want a massive tensor, could set to index 4
                            if (entry['_cutDirection']) > 3:
                                entry['_cutDirection'] = 4
                            label[int((entry['_time'] % 1) * samplerate), entry['_lineIndex'], entry['_lineLayer'], entry['_type'], entry['_cutDirection']] = 1
                        self.items.append(entry)
                            
    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

def main(cuda=torch.cuda.is_available(), gpu=0):
    device = torch.device('cuda:' + str(gpu) if cuda else "cpu")
    data = audio('./samples')
    print(len(data))


if __name__ == "__main__":
    main()
