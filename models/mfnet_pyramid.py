import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter  # 用于进行可视化
from torchviz import make_dot

class Mfnet_pyramid(nn.Module):
    def __init__(self):
        super(Mfnet_pyramid, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size= 3, padding=1)

        self.pyramid1 = torch.nn.Conv2d(20, 10, kernel_size= 3, padding=1)
        self.pyramid2 = torch.nn.Conv2d(20, 10, kernel_size= 3, padding=1)
        self.pyramid3 = torch.nn.Conv2d(20, 10, kernel_size= 3, padding=1)

        self.conv2 = torch.nn.Conv2d(30, 30, kernel_size=3, padding=1)

        self.conv3 = torch.nn.Conv2d(30, 30, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(30, 30, kernel_size=3, padding=1)

        self.conv5 = torch.nn.Conv2d(30, 20, kernel_size=5)
        self.conv6 = torch.nn.Conv2d(20, 10, kernel_size=3)

        self.fc = torch.nn.Linear(640, 10)

        self.Max_pooling = torch.nn.MaxPool2d(2)
        self.Max_pooling4 = torch.nn.MaxPool2d(4)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        p1 = F.relu(self.pyramid1(x))
        p2 = F.relu(self.pyramid2(self.Max_pooling(x)))
        p2 = nn.functional.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = F.relu(self.pyramid3(self.Max_pooling4(x)))
        p3 = nn.functional.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)

        x = torch.cat((p1,p2,p3),dim=1)

        x = F.relu(self.conv2(x))
        x = self.Max_pooling(x)

        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x1))
        x = x + x2

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


# net = Mfnet_pyramid()
# in_channels = 1
# width, height = 28, 28
# batch_size = 2
# ino = torch.randn(batch_size, in_channels, width, height)
# output = net(ino)
#
# # 1. 来用tensorflow进行可视化
# with SummaryWriter("./log", comment="sample_model_visualization") as sw:
#     sw.add_graph(net, ino)
#
# # 2. 保存成pt文件后进行可视化
# torch.save(net, "./log/modelviz.pt")