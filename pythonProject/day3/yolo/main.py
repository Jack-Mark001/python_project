# author: jkisme
from ultralytics import YOLO

def test1():

    model = YOLO("./yolo11s.pt")

    results = model("D:/code/resource/images/hog1.jpg")

    results[0].show()

def test2():
    # 只需修改数据集配置，一行命令启动训练
    model = YOLO("./yolo11s.pt")
    model.train(
        data="./config.yaml",  # 数据集配置文件路径
        epochs=10,  # 训练轮数
        batch=4,  # batch大小
        imgsz=640,  # 输入图片尺寸
        device=0,   # 不写默认使用cpu，要使用GPU
        workers=4,  # 加载数据线程数
        amp=True,  # 关闭混合精度训练（CPU环境建议关闭）
        exist_ok=True,  # 允许覆盖上次训练结果
        cache=False,  # 不缓存数据集到内存
        rect=True  # 矩形训练，减少无效填充
    )

if __name__ == '__main__':
    test2()
