from ultralytics import YOLO
import cv2
model = YOLO("yolov8s.pt") # 원하는 크기 모델 입력(n ~ x)

result = model.predict("./test.jpeg", save=True, conf=0.5)
plots = result[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()
