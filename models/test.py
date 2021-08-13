import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter  # 用于进行可视化
from torchviz import make_dot


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
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


net = Mfnet()

in_channels = 1
width, height = 28, 28
batch_size = 2
ino = torch.randn(batch_size, in_channels, width, height)
output = net(ino)
print(output)

# # 1. 来用tensorflow进行可视化
# with SummaryWriter("./log", comment="sample_model_visualization") as sw:
#     sw.add_graph(net, ino)
#
# # 2. 保存成pt文件后进行可视化
# torch.save(net, "./log/modelviz.pt")

# # 3. 使用graphviz进行可视化
# out = net(ino)
# g = make_dot(out)
# g.render('Mfnet', view=False)  # 这种方式会生成一个pdf文件

