from dataset import fashion_dataset
from models import mfnet
import torch.optim as optim
import torch.nn as nn
import torch
import datetime

# 加载训练集和测试集
batch_size = 8
datapath = "D:\Py_project\My_fashionmnist\data"
mnist_train, mnist_test = fashion_dataset.load_data_fashion_mnist(datapath, batch_size)

# 实例化模型并使用GPU
model = mfnet.Mfnet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_sample(sample):
    """
    训练一个batch
    :param sample: 一个batch
    :return: loss和outputs
    """
    model.train()
    inputs,targets = sample
    inputs, targets = inputs.to(device),targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs,targets)
    loss.backward()
    optimizer.step()
    return loss,outputs

def test_sample(sample):
    """
    测试一个batch
    :param sample: 一个batch
    :return: batch的样本数、正确数
    """
    model.eval()
    inputs,targets = sample
    inputs, targets = inputs.to(device),targets.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, dim=1)
    correct = (predicted == targets).sum().item()
    return targets.size(0),correct

def test():
    """
    测试
    :return: 无
    """
    correct = 0
    total = 0
    for data in mnist_test:
        t,c = test_sample(data)
        correct += c
        total += t
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))

def train(epoch_num, save_path):
    """
    训练
    :param epoch_num: 训练epoch数
    :param save_path: 模型保存路径
    :return: 训练完一个epoch保存模型
    """
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i,data in enumerate(mnist_train, 0):
            loss,outputs = train_sample(data)
            running_loss += loss.item()
            if i%500 == 499:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
        test()
        save_model(save_path)

def save_model(save_path):
    """
    保存模型
    :param save_path: 保存模型路径，示例如下
    :return: 
    """
    # save_path = "D:\Py_project\My_fashionmnist\checkpoints\Mfnet"
    time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(time,'%Y%m%d%H%M%S')
    save_path = save_path + '\\'+ time_str + '.pth'
    print(save_path)
    torch.save(model.state_dict(), save_path)

def load_model(load_path):
    """
    加载模型
    :param load_path: 
    :return: 
    """
    # load_path = 'D:\Py_project\My_fashionmnist\checkpoints\Mfnet\202108121615.pth'
    model.load_state_dict(torch.load(load_path))

if __name__ == '__main__':
    save_path = "D:\Py_project\My_fashionmnist\checkpoints\Mfnet"
    load_path = "D:\Py_project\My_fashionmnist\checkpoints\Mfnet\\20210812175346.pth"
    resume = True
    # 若resume为true,则加载模型继续上次的训练
    if(resume == True):
        load_model(load_path)

    epoch = 2
    train(epoch, save_path)