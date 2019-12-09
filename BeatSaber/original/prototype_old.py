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


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc1 = nn.Linear(88200, 22050)
        self.fc2 = nn.Linear(22050, 8820)
        self.fc3 = nn.Linear(8820, 5880)
        self.fc4 = nn.Linear(5880, 4410)
        self.fc5 = nn.Linear(4410, 2200)
        self.fc6 = nn.Linear(2200, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 256)
        self.fc9 = nn.Linear(256, 148)

    def forward(self, x):
        x = x.view(-1, 88200)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = x.view(-1, 1, 148)
        return x

class audio(Dataset):
    def __init__(self, directory):
        self.items = []

        files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
        # TODO remove this, just for prototyping
        files=files[:10]
        for file in files:
            with open(file, 'rb') as f:
                ## NOTE For reconstruction, this data needs to be added to *items*
                sample = pkl.load(f)
                #name = sample['name'] # this is needed for reconstruction of the song
                #name = sample['name']
                #time = sample['time']
                window = sample['window']
                label = sample['label']
                self.items.append((torch.tensor(np.expand_dims(window, axis=0)).float(), torch.tensor(np.expand_dims(label, axis=0)).float()))
                            
    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


def main(cuda=torch.cuda.is_available(), gpu=0):
    device = torch.device('cuda:' + str(gpu) if cuda else "cpu")
    print(device)
    data = audio('./samples_window')
    print(len(data))

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    data_loader = torch.utils.data.DataLoader(data,batch_size=2,shuffle=True, **kwargs)

    model = network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss(reduction='sum')

    for epoch in range(10):
        model.train()
        for batch_idx, (data, target), in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        test_loss = 0
        model.eval()
        for batch_idx, (data, target), in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)

        print('Epoch:', epoch, 'Loss:', test_loss)

if __name__ == "__main__":
    main(cuda=False)
