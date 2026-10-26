import cv2
from ultralytics import YOLO

# 1. 加载模型
model = YOLO("./runs/detect/train/weights/best.pt")

# 2. 读取图片并推理（换成你测试图的绝对路径）
img_path = r"F:/AI/data/ccpd_green/test/04-91_254-145&472_529&567-529&567_164&552_145&472_529&485-0_0_3_25_24_24_24_29-148-355.jpg"
results = model(img_path)

# 3. 获取带有检测框的 numpy 数组 (BGR格式)
annotated_img = results[0].plot()

# 4. 按比例缩放图片，防止图片太大超出屏幕
h, w = annotated_img.shape[:2]
max_size = 1000  # 限制最大长宽为 1000 像素
scale = min(1.0, max_size / max(h, w))
if scale < 1.0:
    annotated_img = cv2.resize(annotated_img, (int(w * scale), int(h * scale)))

# 5. 显示图片，按任意键关闭
cv2.imshow("YOLO License Plate Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()