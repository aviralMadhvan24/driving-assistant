from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

image_path = "data/rgb/064288.png"

results = model(image_path)

annotated = results[0].plot()

cv2.imshow("YOLO Object Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()