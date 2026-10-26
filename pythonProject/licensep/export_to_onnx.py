from ultralytics import YOLO

# 1. 加载你训练好的 best.pt
model = YOLO("./runs/detect/train/weights/best.pt")

# 2. 一键导出为 ONNX
# imgsz=640 必须与你训练时的尺寸保持一致
# dynamic=True 允许动态 batch 大小
model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)

print("ONNX 导出成功！")