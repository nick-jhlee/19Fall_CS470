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

# Is CUDA available?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import dataset
train_set = GTZANSpecgram('metadata_train.csv')
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# For showing images
'''
# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
'''

#net = Model_CNN()
net = Model_CNN_Mel()
net.to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0015, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        images, genres = data[0].to(device), data[1].to(device)
        genres = torch.tensor(genres, dtype=torch.long).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images).to(device)
        loss = criterion(outputs, genres).to(device)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# Save the trained network
pathlib.Path('net/').mkdir(parents=True, exist_ok=True)
PATH = 'net/net.pth'
torch.save(net.state_dict(), PATH)

