# author: jkisme
from ultralytics import YOLO

def train():
    model = YOLO("./yolo11s.pt")
    model.train(
        data="./config.yaml",  # 数据集配置文件路径
        epochs=20,  # 训练轮数
        batch=8,  # batch大小
        imgsz=640,  # 输入图片尺寸
        device=0,  # 不写默认使用cpu，要使用GPU
        workers=2,  # 加载数据线程数# Windows 下建议 0 或 2
        amp=True,  # 关闭混合精度训练（CPU环境建议关闭）
        exist_ok=True,  # 允许覆盖上次训练结果
        cache=False,  # 不缓存数据集到内存
        rect=False,  # 矩形训练，减少无效填充
        close_mosaic=10,  # 最后 10 轮关闭 Mosaic 数据增强，大幅降低显存，提升精度
    )

if __name__ == '__main__':
    train()
