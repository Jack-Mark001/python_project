# author: jkisme
import torch
import torch.nn as nn

# 模型构建   nn.Module 是父类（继承）
class ImageClassification(nn.Module):
    # 定义网络结构  __init__ 是构造函数（创建对象时自动执行）
    def __init__(self):
        # super().__init__() 调用父类的构造函数，初始化 nn.Module 内部机制
        super(ImageClassification, self).__init__()
        # 定义网络层：卷积层+池化层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层
        self.linear1 = nn.Linear(in_features=576, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)


    # 定义前向传播
    def forward(self, x):
        # 卷积+relu+池化
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        # 卷积+relu+池化
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # 将特征图做成以为向量的形式：相当于特征向量
        x = x.reshape(x.size(0), -1)
        # 全连接层
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # 返回输出结果
        return self.out(x)