from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator
import cv2

class PoseMonitoring(BaseSolution):
    def __init__(self, **kwargs):
        """Initializes AIGym for workout monitoring using pose estimation and predefined angles."""
        # Check if the model name ends with '-pose'
        if "model" in kwargs and "-pose" not in kwargs["model"]:
            kwargs["model"] = "yolo11n-pose.pt"
        elif "model" not in kwargs:
            kwargs["model"] = "yolo11n-pose.pt"

        super().__init__(**kwargs)
        self.count = []  # List for counts, necessary where there are multiple objects in frame
        self.angle = []  # List for angle, necessary where there are multiple objects in frame
        self.stage = []  # List for stage, necessary where there are multiple objects in frame

        # Extract details from CFG single time for usage later
        self.initial_stage = None
        self.up_angle = float(self.CFG["up_angle"])  # Pose up predefined angle to consider up pose
        self.down_angle = float(self.CFG["down_angle"])  # Pose down predefined angle to consider down pose
        self.kpts = self.CFG["kpts"]  # User selected kpts of workouts storage for further usage

    def draw_skeleton(self, im0, keypoints, conf_thres=0.5):
        # Danh sách các kết nối (skeleton structure) giữa các điểm keypoints
        POSE_SKELETON = {
            (0, 1): (147, 20, 255),  # Mũi - Mắt phải
            (0, 2): (255, 255, 0),   # Mũi - Mắt trái
            (1, 3): (147, 20, 255),  # Mắt phải - Tải phải
            (2, 4): (255, 255, 0),   # Mắt trái - Tai trái
            # (0, 5): (147, 20, 255),  # Mũi - Vai phải
            # (0, 6): (255, 255, 0),   # Mũi - Vai trái
            (5, 7): (147, 20, 255),  # Vai phải - Khuỷu tay phải
            (7, 9): (147, 20, 255),  # Khuỷu tay phải - Cổ tay phải
            (6, 8): (255, 255, 0),   # Vai trái - Khuỷu tay trái
            (8, 10): (255, 255, 0),  # Khuỷu tay trái - Cổ tay trái
            (5, 6): (0, 255, 255),   # Vai phả - Vai trái
            (5, 11): (147, 20, 255), # Vai phải - Hông phải
            (6, 12): (255, 255, 0),  # Left Shoulder - Hông trái
            (11, 12): (0, 255, 255), # Hông phải - Hông trái
            (11, 13): (147, 20, 255),# Hông phải - Gối phải
            (13, 15): (147, 20, 255),# Gối phải - Cổ chân phải
            (12, 14): (255, 255, 0), # Hông trái - Gối trái
            (14, 16): (255, 255, 0)  # Gối trái - Cổ chân trái
        }

        # Vẽ keypoints
        for i in range(len(keypoints)):
            x, y, conf = keypoints[i]
            if conf > conf_thres:
                cv2.circle(im0, (int(x), int(y)), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        
        # Vẽ các đường nối (skeleton) giữa các điểm keypoints
        for (start_idx, end_idx), color in POSE_SKELETON.items():
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            start_x, start_y, _ = start_point
            end_x, end_y, _ = end_point
            if start_point[2] > conf_thres and end_point[2] > conf_thres:
                cv2.line(im0, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 255), 1)  # Vẽ đường nối màu vàng
                # cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, 2)  # Vẽ đường nối theo màu định nghĩa

    def monitor(self, im0):
        # Extract tracks
        tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"])[0]
        # keypoints = tracks.keypoints.data.cpu().numpy()
        # self.draw_skeleton(im0, keypoints, confidence_threshold=0.5)

        if tracks.boxes.id is not None:
            for ind, k in enumerate(tracks.keypoints.data.cpu().numpy()):
                self.draw_skeleton(im0, keypoints=k, conf_thres=0.5)

            boxes = tracks[0].boxes.xywh.cpu().numpy()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()
            confidences = tracks[0].boxes.conf.cpu().numpy()
            confidence_threshold = 0.25
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                if conf < confidence_threshold:
                    continue
                x, y, w, h = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)

                # Draw the bounding box
                im0 = cv2.rectangle(im0, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

                # Label the bounding box with track ID
                label = f"ID: {track_id} ({conf:.2f})"
                cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)


        # if tracks.boxes.id is not None:
        #     # Extract and check keypoints
        #     if len(tracks) > len(self.count):
        #         new_human = len(tracks) - len(self.count)
        #         self.angle += [0] * new_human
        #         self.count += [0] * new_human
        #         self.stage += ["-"] * new_human

        #     # Initialize annotator
        #     self.annotator = Annotator(im0, line_width=self.line_width)

        #     # Enumerate over keypoints
        #     for ind, k in enumerate(reversed(tracks.keypoints.data)):
        #         print('keypoint', k.cpu().numpy())
        #         # Get keypoints and estimate the angle
        #         # kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
        #         # self.angle[ind] = self.annotator.estimate_pose_angle(*kpts)
        #         # im0 = self.annotator.draw_specific_points(k, self.kpts, radius=self.line_width * 3)

        #         # # Determine stage and count logic based on angle thresholds
        #         # if self.angle[ind] < self.down_angle:
        #         #     if self.stage[ind] == "up":
        #         #         self.count[ind] += 1
        #         #     self.stage[ind] = "down"
        #         # elif self.angle[ind] > self.up_angle:
        #         #     self.stage[ind] = "up"

        #         # # Display angle, count, and stage text
        #         # self.annotator.plot_angle_and_count_and_stage(
        #         #     angle_text=self.angle[ind],  # angle text for display
        #         #     count_text=self.count[ind],  # count text for workouts
        #         #     stage_text=self.stage[ind],  # stage position text
        #         #     center_kpt=k[int(self.kpts[1])],  # center keypoint for display
        #         # )

        # # self.display_output(im0)
        return im0
