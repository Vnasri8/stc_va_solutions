import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-pose.pt')

# Đọc ảnh
# image = cv2.imread('pose-img.jpg')
image = cv2.imread('people.jpg')
if image is None:
    print("Không thể đọc ảnh.")
    exit()

# Chạy mô hình YOLOv8 Pose để phát hiện keypoints
results = model(image)

# Danh sách các kết nối (skeleton structure) giữa các điểm keypoints
POSE_SKELETON = {
    (0, 1): (147, 20, 255),  # Mũi - Mắt phải
    (0, 2): (255, 255, 0),   # Mũi - Mắt trái
    (1, 3): (147, 20, 255),  # Mắt phải - Tải phải
    (2, 4): (255, 255, 0),   # Mắt trái - Tai trái
    (0, 5): (147, 20, 255),  # Mũi - Vai phải
    (0, 6): (255, 255, 0),   # Mũi - Vai trái
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

# Vẽ keypoints và skeleton lên ảnh
for result in results:
    # Mỗi kết quả (result) có thể chứa một hoặc nhiều đối tượng (người)
    keypoints = result.keypoints.data.cpu().numpy()  # Chuyển sang numpy nếu sử dụng GPU

    # Vẽ keypoints cho từng người
    for person_keypoints in keypoints:

        # Vẽ keypoints
        for i in range(len(person_keypoints)):
            x, y, conf = person_keypoints[i]
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Vẽ các đường nối (skeleton) giữa các điểm keypoints
        for (start_idx, end_idx), color in POSE_SKELETON.items():
            start_point = person_keypoints[start_idx]
            end_point = person_keypoints[end_idx]

            start_x, start_y, _ = start_point
            end_x, end_y, _ = end_point
            if start_point[2] > 0.5 and end_point[2] > 0.5:
                cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 255), 2)  # Vẽ đường nối màu vàng
                # cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, 2)  # Vẽ đường nối theo màu định nghĩa

# Hiển thị kết quảq
cv2.imshow("Pose Detection with Skeleton", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
