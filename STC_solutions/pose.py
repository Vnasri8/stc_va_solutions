from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator
import cv2
import torch
import time


class PoseMonitoring(BaseSolution):
    def __init__(self, **kwargs):
        if "model" in kwargs and "-pose" not in kwargs["model"]:
            kwargs["model"] = "yolo11n-pose.pt"
        elif "model" not in kwargs:
            kwargs["model"] = "yolo11n-pose.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(**kwargs)

        self.model.to(self.device)
        print(f'Model loaded on {self.device}')

    def monitor(self, im0):
        start_time = time.time()
        
        results  = self.model.track(source=im0, persist=True)
        # print(results[0].boxes)
        if results[0] is None or results[0].boxes is None:
            print("No objects detected in the frame.")
            return im0
    
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = im0.copy()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)

            color = (0, 255, 0)
            thickness = 4
            annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # label = f"{CLASSES[class_id]} ({confidence:.2f})"
            label = f"ID: {track_id}"
            # cv2.putText(annotated_frame, label, (x1, y1 - 10), font, 2, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.4f} seconds")
        self.display_output(annotated_frame)
        return annotated_frame