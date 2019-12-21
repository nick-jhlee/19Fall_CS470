import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch
import pathlib
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from gtzanspecgram import GTZANSpecgram
from gtzanmelspec import GTZANMel
from cnn import Model_CNN, Model_CNN_Mel

PATH = 'net/net.pth'

# Is CUDA available?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#net = Model_CNN()
net = Model_CNN_Mel()
net.to(device)
net.load_state_dict(torch.load(PATH))

# Load the test set
test_set = GTZANSpecgram('metadata_test.csv')
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, genres = data[0].to(device), data[1].to(device)
        genres = torch.tensor(genres, dtype=torch.long).to(device)
        outputs = net(images).to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += genres.size(0)
        correct += (predicted == genres).sum().item()

print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))

# TODO: Would be nice to make a confusion matrix