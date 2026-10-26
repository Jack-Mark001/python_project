# author: jkisme
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
BATCH_SIZE = 128

#训练函数
def train(model,train_dataset):
    criterion = nn.CrossEntropyLoss() # 构建损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # 构建优化方法
    epoch = 100   # 训练轮数
    for epoch_idx in range(epoch):
        # 构建数据加载器
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        sam_num = 0   # 样本数量
        total_loss = 0.0    # 损失总和
        start = time.time() # 开始时间
        # 遍历数据进行网络训练
        for x, y in dataloader:
            output = model(x)
            loss = criterion(output, y) # 计算损失
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 参数更新
            total_loss += loss.item() # 统计损失和
            sam_num += 1
        print('epoch:%2s loss:%.5f time:%.2fs' %(epoch_idx + 1,total_loss / sam_num,time.time() - start))
    # 模型保存
    torch.save(model.state_dict(), 'data/image_classification.pth')