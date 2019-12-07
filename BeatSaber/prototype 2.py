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
        self.conv1 = nn.Conv2d(1, 10, (3,2), 2)
        self.conv2 = nn.Conv2d(10, 10, (3,1), 2)
        self.conv3 = nn.Conv2d(10, 10, (3,1), 2)
        self.conv4 = nn.Conv2d(10, 10, (3,1), 2)
        self.conv5 = nn.Conv2d(10, 10, (3,1), 2)
        self.conv6 = nn.Conv2d(10, 10, (3,1), 2)
        self.conv7 = nn.Conv2d(10, 10, (3,1), 2)
        self.conv8 = nn.Conv2d(10, 10, (3,1), 2)
        self.conv9 = nn.Conv2d(10, 10, (3,1), 2)
        self.conv10 = nn.Conv2d(10, 10, (3,1), 2)
        self.fc1 = nn.Linear(420, 300)
        self.fc2 = nn.Linear(300, 200) 
        self.fc3 = nn.Linear(200, 148)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x = torch.relu(self.conv8(x))
        x = torch.relu(self.conv9(x))
        x = torch.relu(self.conv10(x))
        x = x.view(-1, 1, 420)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.view(-1, 148)
        return x

class audio(Dataset):
    def __init__(self, directory):
        self.items = []

        files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
        # TODO remove this, just for prototyping
        # files=files[:10]
        for file in files:
            self.items.append(file)
                            
    def __getitem__(self, index):
        file = self.items[index]
        with open(file, 'rb') as f:
                sample = pkl.load(f)
                window = sample['window']
                label = sample['label']
                return (torch.tensor(np.expand_dims(window, axis=0)).float(), torch.tensor(label).long())

    def __len__(self):
        return len(self.items)

# class audio(Dataset):
#     def __init__(self, directory):
#         self.items = []

#         files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
#         # TODO remove this, just for prototyping
#         # files=files[:10]
#         for file in files:
#             self.items.append(file)
                            
#     def __getitem__(self, index):
#         file = self.items[index]
#         with open(file, 'rb') as f:
#                 sample = pkl.load(f)
#                 window = sample['window']
#                 label = sample['label']
#                 return (torch.tensor(np.expand_dims(np.sum(window, axis=1), axis=0)).float(), torch.tensor(np.expand_dims(label, axis=0)).float())

#     def __len__(self):
#         return len(self.items)


def main(cuda=torch.cuda.is_available(), gpu=0):
    device = torch.device('cuda:' + str(gpu) if cuda else "cpu")
    print(device)
    data = audio('./samples_window')
    print(len(data))

    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    data_loader = torch.utils.data.DataLoader(data,batch_size=256,shuffle=True, **kwargs)

    model = network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        test_loss = 0
        for batch_idx, (data, target), in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            #test_loss += loss
            print('Epoch', epoch, 'Batch', batch_idx, '/', len(data_loader), 'Loss', loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        
        # test_loss = 0
        # model.eval()
        # for batch_idx, (data, target), in enumerate(data_loader):
        #     data, target = data.to(device), target.to(device)
        #     output = model(data)
        #     test_loss += criterion(output, target)

        # print('Epoch:', epoch, 'Loss:', test_loss)

        #print('Epoch', epoch, 'Loss', test_loss.cpu().detach().numpy())

        torch.save(model.state_dict(), 'model' + str(gpu) + '_' + str (epoch) + '.pt')

if __name__ == "__main__":
    main(cuda=True)
