# author: jkisme
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor



# 1. 数据集基本信息
def create_dataset():
    # 加载数据集:训练集数据和测试数据
    train = CIFAR10(root='data', train=True, download=True,  transform=Compose([ToTensor()]))
    valid = CIFAR10(root='data', train=False, download=True,  transform=Compose([ToTensor()]))
    #返回数据集结果
    return train, valid
if __name__ == '__main__':
    # 数据集加载
    train_dataset,valid_dataset = create_dataset()
    print(train_dataset)
    print(valid_dataset)