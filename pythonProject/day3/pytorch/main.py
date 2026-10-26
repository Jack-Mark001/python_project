# author: jkisme
import torch

from  ImageClassification import ImageClassification
from load_data import *
from train_model import *

# 加载数据集
train_dataset, test_dataset = create_dataset()

def test():
    # 构建数据加载器
    dataloader= DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
    # 加载模型加载训练好的权重
    model=ImageClassification()
    model.load_state_dict(torch.load('data/image_classification.pth'))
    model.eval()
    # 计算精度
    total_correct=0
    total_samples=0

    # 遍历每个batch数据
    for x,y in dataloader:
        output=model(x)
        total_correct+= (torch.argmax(output,dim=1)==y).sum()
        total_samples+=len(y)

    # 打印精度
    print('Acc:%.4f' %(total_correct/total_samples))

test()