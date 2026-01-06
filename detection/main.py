import cv2
import numpy as np
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select video",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    exit()


CONF_THRESHOLD = 0.4 #speed change
IOU_THRESHOLD = 0.2 #collision
CENTER_COLLISION_DIST = 40     # pixels - confidence score calculated on this
SPEED_DROP_RATIO = 0.6
MAX_JUMP_DISTANCE = 120 #flyaway
ACCIDENT_WINDOW = 8 #collision time
VEHICLE_CLASSES = [2, 3, 5, 7]


model = YOLO("yolov8n.pt")


def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

cap = cv2.VideoCapture(video_path)

prev_positions = {}
prev_speeds = {}
prev_ids = set()

collision_timer = 0
collision_seen = False
speed_drop_seen = False
glitch_seen = False
yolo_conf_history = []

accident_detected = False
confidence = 0.0

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = max(now - prev_time, 1e-3)
    prev_time = now

    results = model.track(frame, persist=True, conf=CONF_THRESHOLD)

    boxes, ids, confs = [], [], []

    if results[0].boxes.id is not None:
        for box, cls, tid, conf in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
            results[0].boxes.id.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy()
        ):
            if int(cls) in VEHICLE_CLASSES:
                boxes.append(box)
                ids.append(int(tid))
                confs.append(float(conf))

    #collision detection 

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            overlap = iou(boxes[i], boxes[j])

            cx1 = (boxes[i][0] + boxes[i][2]) / 2
            cy1 = (boxes[i][1] + boxes[i][3]) / 2
            cx2 = (boxes[j][0] + boxes[j][2]) / 2
            cy2 = (boxes[j][1] + boxes[j][3]) / 2

            center_dist = np.hypot(cx1 - cx2, cy1 - cy2)

            if overlap > IOU_THRESHOLD or center_dist < CENTER_COLLISION_DIST:
                collision_seen = True
                collision_timer = ACCIDENT_WINDOW
                yolo_conf_history.append((confs[i] + confs[j]) / 2)
                break

    #speed + glitch

    current_ids = set(ids)

    for box, tid in zip(boxes, ids):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        if tid in prev_positions:
            px, py = prev_positions[tid]
            dist = np.hypot(cx - px, cy - py)
            speed = dist / dt
            prev_speed = prev_speeds.get(tid, speed)

            if speed < prev_speed * SPEED_DROP_RATIO:
                speed_drop_seen = True

            if dist > MAX_JUMP_DISTANCE:
                glitch_seen = True

            prev_speeds[tid] = speed

        prev_positions[tid] = (cx, cy)

    if prev_ids - current_ids:
        glitch_seen = True

    prev_ids = current_ids

#final decision

    if collision_timer > 0:
        collision_timer -= 1

        if collision_seen and (speed_drop_seen or glitch_seen):
            accident_detected = True

            confidence = (
                0.4 +                                  # collision
                (0.3 if speed_drop_seen else 0) +
                (0.2 if glitch_seen else 0) +
                (min(np.mean(yolo_conf_history), 1.0) * 0.1)
            )

            confidence = round(min(confidence, 1.0), 3)
            break

cap.release()

print("Accident Detected:", accident_detected)
print("Confidence:", confidence)
