# author: jkisme

import cv2
from ultralytics import YOLO

def test1():

    model = YOLO("./runs/detect/train/weights/best.pt")

    results = model("D:/code/resource/images/crowd.jpg")

    img = results[0].plot()  # 返回带框的图片数组
    # 按比例缩小，让长边不超过 900 像素（可按你的屏幕调整）
    h, w = img.shape[:2]
    scale = min(1.0, 900 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)  # 按任意键关闭窗口
    cv2.destroyAllWindows()

test1()