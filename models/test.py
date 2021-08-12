import torch
import torch.nn as nn
import torch.nn.functional as F

class Mfnet(nn.Module):
    def __init__(self):
        super(Mfnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size= 3)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size= 5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        print(x.shape)
        x = F.relu(self.pooling(self.conv2(x)))
        print(x.shape)
        x = x.view(batch_size, -1)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        return x


net = Mfnet()

in_channels = 1
width, height = 28, 28
batch_size = 2
ino = torch.randn(batch_size, in_channels, width, height)
output = net(ino)

print(output)

