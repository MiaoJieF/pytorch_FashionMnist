import torchvision
from torchvision import datasets, transforms
from matplotlib import pyplot
from torch.utils.data import DataLoader
import numpy

def load_data_fashion_mnist(path, batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = path, train = True, transform = trans, download = False)
    mnist_test = torchvision.datasets.FashionMNIST(root = path, train = False, transform = trans, download = False)
    return (DataLoader(mnist_train,batch_size,shuffle=True, num_workers=0),
            DataLoader(mnist_test,batch_size,shuffle=False, num_workers=0))

def get_fashion_mnist_labels(index):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return text_labels[index]

def show_picture(images, labels, row, col):
    """
    显示图片
    :param images: 一个batch
    :param labels: 标签
    :param row: 行数
    :param col: 列数
    :return: 
    """
    index = 1
    for i in images:
        i = i.squeeze(dim=0) # 去掉第一个维度，变成二维
        image = i.numpy()
        pyplot.subplot(row, col, index)
        l = labels[index-1]
        label = get_fashion_mnist_labels(l)
        pyplot.imshow(image)
        pyplot.title(label)
        index = index + 1
    pyplot.show()

# # 显示图片示例代码
# datapath = "D:\Py_project\pytorch_FashionMnist\data"
# mnist_train,mnist_test = load_data_fashion_mnist(datapath, batch_size=18)
# print(len(mnist_train),len(mnist_test))
# dataiter = iter(mnist_train)
# images, labels = dataiter.next()
# print(len(images))
# show_picture(images,labels,3,6)
