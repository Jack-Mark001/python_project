# author: jkisme
from load_data import create_dataset
from ImageClassification import ImageClassification
from train_model import *

if __name__ == '__main__':
    # 数据集加载
    train_dataset, valid_dataset = create_dataset()
    # 模型实例化
    model = ImageClassification()
    # 模型训练
    train(model,train_dataset)
    print("模型训练完成！")
