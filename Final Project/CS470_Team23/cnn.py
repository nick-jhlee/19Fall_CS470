import torch.nn as nn
import torch.nn.functional as F

class Model_CNN(nn.Module):
    def __init__(self):
        super(Model_CNN, self).__init__()
        # height and width dimensions are different
        self.conv1 = nn.Conv2d(3, 34, (2, 14)).cuda()
        self.pool = nn.MaxPool2d(2, 2).cuda()
        self.conv2 = nn.Conv2d(34, 34, (2, 14)).cuda()
        self.conv3 = nn.Conv2d(34, 64, 5).cuda()
        self.conv4 = nn.Conv2d(64, 96, 5).cuda()
        self.conv5 = nn.Conv2d(96, 128, 5).cuda()
        self.fc1 = nn.Linear(128 * 8 * 6, 200).cuda()
        self.fc2 = nn.Linear(200, 128).cuda()
        self.fc3 = nn.Linear(128, 10).cuda()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        #print(x.shape)
        #print(x.size(0))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Model_CNN_Mel(nn.Module):
    def __init__(self):
        super(Model_CNN_Mel, self).__init__()
        # height and width dimensions are different
        self.conv1 = nn.Conv2d(3, 16, (3, 7)).cuda()
        self.conv2 = nn.Conv2d(16, 32, (3, 7)).cuda()
        self.pool1 = nn.MaxPool2d(3, 7).cuda()
        self.conv3 = nn.Conv2d(32, 64, 3).cuda()
        self.conv4 = nn.Conv2d(64, 96, 3).cuda()
        self.pool2 = nn.MaxPool2d(3, 3).cuda()
        self.conv5 = nn.Conv2d(96, 128, 3).cuda()
        self.fc1 = nn.Linear(128 * 8 * 8, 128).cuda()
        self.fc2 = nn.Linear(128, 64).cuda()
        self.fc3 = nn.Linear(64, 10).cuda()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        #print(x.shape)
        #print(x.size(0))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
