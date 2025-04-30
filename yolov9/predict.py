import cv2
from ultralytics import YOLO
img_pth = "train/images/000001_jpg.rf.b172f9997bfb7e4933a72e5ae7bc638a.jpg"
model = YOLO("runs/detect/train3/weights/best.pt")
results = model(source=img_pth)
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)

