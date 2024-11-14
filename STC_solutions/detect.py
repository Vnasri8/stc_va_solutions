from time import time

import cv2
import torch
from ultralytics import YOLO
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class ObjectDetection(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_classes = self.CFG["target_classes"]

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

    def predict(self, im0):
        """Run prediction using a YOLO model for the input image `im0`."""
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        """Displays the FPS on an image `im0` by calculating and overlaying as white text on a black rectangle."""
        self.end_time = time()
        fps = 1 / round(self.end_time - self.start_time, 2)
        text = f"FPS: {int(fps)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(
            im0,
            (20 - gap, 70 - text_size[1] - gap),
            (20 + text_size[0] + gap, 70 + gap),
            (0, 255, 0),
            -1,
        )
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        """Plots bounding boxes on an image given detection results; returns annotated image and class IDs."""
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            if self.target_classes and names[int(cls)] not in self.target_classes:
                continue
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def detect(self, im0):
        self.start_time = time()
        results = self.predict(im0)
        im0, class_ids = self.plot_bboxes(results, im0)
        self.display_fps(im0)
        # self.display_output(im0)
        return im0