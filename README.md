# pytorch_FashionMnist
 Pytorch based network for FashionMnist

本项目为基于Pytorch的对于FashionMnist数据集的基本代码框架，包括数据加载、显示数据、模型定义、训练、测试、保存模型、加载模型等基本示例。

****

### 文件夹结构说明

```
+---checkpoints
|   \---Mfnet
|       +---pretrain_model.pth
|   \---Mfnet_pyramid
|       +---pretrain_model.pth
+---data
|   \---FashionMNIST
|       +---processed
|       \---raw
+---dataset
|   \---fashion_dataset.py
+---models
|   +---log
|   \---mfnet.py
|   \---mfnet_pyramid.py
main.py
```

cheeckpoints文件夹：保存训练模型

data文件夹：数据集

dataset文件夹：加载数据集的py文件

models文件夹：网络模型类

main.py：训练和测试的代码

****

### 使用说明

修改main.py中的三个路径即可运行

```
batchsize：按需修改
epoch：按需修改
datapath：数据集的存放路径
save_path：模型的保存文件夹路径
load_path：resume设置为True时，修改为需加载的模型的路径
```

****

### 示例模型说明

```
Mfnet：非常简单的模型，只有两层卷积层，一层全连接层，实验中在测试集上的精度最佳为88.54%
Mfnet_pyramid：使用金字塔特征提取、一个Resnet层和几层卷积层组成，实验中在测试集上的精度最佳为91.98%
```

