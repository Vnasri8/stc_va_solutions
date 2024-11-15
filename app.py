from time import time

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from STC_solutions.detect import ObjectDetection
from STC_solutions.pose import PoseMonitoring
from ultralytics import solutions

# src = "rtsp://admin:Admin123456%2A%40@192.168.8.191:554/cam/realmonitor?channel=1&subtype=0"
# src = "rtsp://admin:Stc%40vielina.com@192.168.8.192:554/cam/realmonitor?channel=1&subtype=0"
src = 0

cap = cv2.VideoCapture(src)
assert cap.isOpened()

solution_id = 2

match solution_id:
    case 1:
        # Solution for person detection
        classes = ['person']
        detector = ObjectDetection(target_classes = classes, model = "yolo11n.pt", show=True, line_width=10)
    case 2:
        # Solutions for pose monitoring
        pose_monitoring = PoseMonitoring(show=True, model="yolo11n-pose.pt", kpts=[6, 8, 10])


while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = cv2.resize(im0, (640, 480))
    match solution_id:
        case 1:
            annotated_frame =  detector.detect(im0)
        case 2:
            annotated_frame =  pose_monitoring.monitor(im0)

    cv2.imshow("STREAMING", annotated_frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()