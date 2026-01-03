import cv2
import time
from ultralytics import YOLO
from shapely.geometry import Point
from detection.tracker import VehicleTracker
# from detection.drowsiness import VideoFrameHandler  # Commented out
from detection.roi import ROAD_ROI, ZEBRA_ROI
from detection.utils import box_center, iou
from detection.risk_engine import compute_risk
import numpy as np

# Global instances for detect_accident
model = None
vehicle_tracker = None
vehicle_memory = {}
drowsy_detector = None

def init_models():
    global model, vehicle_tracker, drowsy_detector
    if model is None:
        model = YOLO("yolov8s.pt")
        vehicle_tracker = VehicleTracker()
        # drowsy_detector = VideoFrameHandler()  # Commented out due to mediapipe issue

def detect_accident(frame):
    init_models()
    
    events = set()
    SPEED_LIMIT = 60
    VEHICLES = ["car", "motorcycle", "bus", "truck"]
    DROWSINESS_THRESHOLDS = {"EAR_THRESH": 0.25, "WAIT_TIME": 2.0}

    # yolo detection
    results = model(frame, verbose=False)

    persons = []
    phones = []
    vehicle_detections = []

    for r in results:
        if r.boxes is None:
            continue

        for b in r.boxes:
            cls = model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            if cls == "person":
                persons.append([x1, y1, x2, y2])

            elif cls == "cell phone":
                phones.append([x1, y1, x2, y2])

            elif cls in VEHICLES:
                vehicle_detections.append([x1, y1, x2, y2, conf, cls])

    # jaywalking
    for p in persons:
        center = Point(box_center(p))
        if ROAD_ROI.contains(center) and not ZEBRA_ROI.contains(center):
            events.add("jaywalking")

    # phone distraction
    for p in persons:
        for ph in phones:
            if iou(p, ph) > 0.05:
                events.add("phone")

    # deepsort tracker && speeding
    tracks = vehicle_tracker.update(vehicle_detections, frame)

    track_boxes = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if track_id in vehicle_memory:
            px, py = vehicle_memory[track_id]
            dist = ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5
            speed = dist * 0.05 * 30  # scaled estimate

            if speed > SPEED_LIMIT:
                events.add("speeding")

        vehicle_memory[track_id] = center
        track_boxes.append([x1, y1, x2, y2])

    # accident detection
    for i in range(len(track_boxes)):
        for j in range(i + 1, len(track_boxes)):
            if iou(track_boxes[i], track_boxes[j]) > 0.15:
                events.add("accident")

    # drowsiness - commented out due to mediapipe issue
    # frame, play_alarm = drowsy_detector.process(frame, thresholds=DROWSINESS_THRESHOLDS)
    # if play_alarm:
    #     events.add("drowsy")

    # risk score
    risk = compute_risk(events)

    accident_detected = "accident" in events
    confidence = risk  # Use risk as confidence

    return accident_detected, confidence

#config

#VIDEO_PATH = "data/demo.mp4"   
VIDEO_PATH = 0 #webcam
SPEED_LIMIT = 60

VEHICLES = ["car", "motorcycle", "bus", "truck"]

DROWSINESS_THRESHOLDS = {
    "EAR_THRESH": 0.25,
    "WAIT_TIME": 2.0
}

#models

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video source")

vehicle_tracker = VehicleTracker()
vehicle_memory = {}     # track_id -> previous center

# drowsy_detector = VideoFrameHandler()  # Commented out

#main loop

if __name__ == "__main__":

    while True:
        ret, frame = cap.read()
        if not ret:
            break

    #main loop

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        events = set()

        #yolo detection

        results = model(frame, verbose=False)

        persons = []
        phones = []
        vehicle_detections = []

        for r in results:
            if r.boxes is None:
                continue

            for b in r.boxes:
                cls = model.names[int(b.cls[0])]
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])

                if cls == "person":
                    persons.append([x1, y1, x2, y2])

                elif cls == "cell phone":
                    phones.append([x1, y1, x2, y2])

                elif cls in VEHICLES:
                    vehicle_detections.append([x1, y1, x2, y2, conf, cls])

        #jaywalking

        for p in persons:
            center = Point(box_center(p))
            if ROAD_ROI.contains(center) and not ZEBRA_ROI.contains(center):
                events.add("jaywalking")
                cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)

        #phone distraction

        for p in persons:
            for ph in phones:
                if iou(p, ph) > 0.05:
                    events.add("phone")
                    cv2.rectangle(frame, (ph[0], ph[1]), (ph[2], ph[3]), (255, 0, 0), 2)

        #deepsort tracker && speeding

        tracks = vehicle_tracker.update(vehicle_detections, frame)

        track_boxes = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track_id in vehicle_memory:
                px, py = vehicle_memory[track_id]
                dist = ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5
                speed = dist * 0.05 * 30  # scaled estimate

                if speed > SPEED_LIMIT:
                    events.add("speeding")
                    cv2.putText(
                        frame,
                        f"SPEEDING ID {track_id}",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            vehicle_memory[track_id] = center
            track_boxes.append([x1, y1, x2, y2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        #accident detection

        for i in range(len(track_boxes)):
            for j in range(i + 1, len(track_boxes)):
                if iou(track_boxes[i], track_boxes[j]) > 0.15:
                    events.add("accident")

        #drowsiness

        frame, play_alarm = drowsy_detector.process(
            frame, thresholds=DROWSINESS_THRESHOLDS
        )

        if play_alarm:
            events.add("drowsy")

        #risk score
        risk = compute_risk(events)

        output = {
            "timestamp": time.time(),
            "events": sorted(events),
            "risk_score": risk
        }
        print(output)

        #display

        cv2.putText(
            frame,
            f"Risk: {risk}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("SMART ROAD SAFETY AI", frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

    #cleanup

    cap.release()
    cv2.destroyAllWindows()
