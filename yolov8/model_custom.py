from ultralytics import YOLO
# import torch

model = YOLO('yolov8s-cls.yaml')
model = YOLO('yolov8s-cls.pt')

model.train(data='/custom_cls/', epochs=20)

#원천 데이터 분류
#person, road,road line

