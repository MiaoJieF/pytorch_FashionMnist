from dataset import fashion_dataset
from models import mfnet
import torch.optim as optim
import torch.nn as nn
import torch
import datetime

batch_size = 8
datapath = "D:\Py_project\My_fashionmnist\data"

mnist_train, mnist_test = fashion_dataset.load_data_fashion_mnist(datapath, batch_size)

net = mfnet.Mfnet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(epoch_num):
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i,data in enumerate(mnist_train, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%500 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in mnist_test:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))

def save_model(save_path):
    # save_path = "D:\Py_project\My_fashionmnist\checkpoints\Mfnet"
    time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(time,'%Y%m%d%H%M%S')
    save_path = save_path + '\\'+ time_str + '.pth'
    print(save_path)
    torch.save(net.state_dict(), save_path)

def load_model(load_path):
    # load_path = 'D:\Py_project\My_fashionmnist\checkpoints\Mfnet\202108121615.pth'
    net = mfnet.Mfnet()
    net.load_state_dict(torch.load(load_path))

if __name__ == '__main__':
    pass