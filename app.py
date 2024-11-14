from time import time

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from STC_solutions.detect import ObjectDetection
from ultralytics import solutions

src = "rtsp://admin:Stc%40vielina.com@192.168.8.192:554/cam/realmonitor?channel=1&subtype=0"
# src = 0
cap = cv2.VideoCapture(src)
assert cap.isOpened()
classes = ['person']
detector = ObjectDetection(target_classes = classes, model = "yolo11n.pt", show=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = cv2.resize(im0, (640, 480))
    annotated_frame =  detector.detect(im0)
    cv2.imshow("STREAMING", annotated_frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()