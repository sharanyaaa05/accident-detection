import cv2
import time
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point

from tracker import VehicleTracker
from drowsiness import VideoFrameHandler
from roi import ROAD_ROI, ZEBRA_ROI, STOP_LINE_ROI
from utils import box_center, iou
from risk_engine import compute_risk

#cctv video config

VIDEO_PATH = 0 #"data/demo.mp4"
FPS_FALLBACK = 25

SPEED_LIMIT = 60          # km/h (approx)
MIN_CRASH_SPEED = 10      # km/h – “impact” must be non‑trivial
ACCIDENT_FRAME_THRESH = 3 # frames with overlap + impact
PERSON_MOVE_THRESH = 8
FLASH_PERIOD = 10         # for flashing red overlay

VEHICLES = ["car", "motorcycle", "bus", "truck"]
SIGNAL_STATE = "RED"      # fixed for demo; could be fed from backend

DROWSINESS_THRESHOLDS = {
    "EAR_THRESH": 0.25,
    "WAIT_TIME":  2.0
}

# Safety: max age for a collision pair without overlap before resetting
COLLISION_DECAY = 15      # frames


#initialization

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video source")

FPS = cap.get(cv2.CAP_PROP_FPS)
FPS = FPS if FPS and FPS > 0 else FPS_FALLBACK

vehicle_tracker = VehicleTracker()
drowsy_detector = VideoFrameHandler()

# Memory
vehicle_pos = {}          # track_id -> (cx, cy)
speed_memory = {}         # track_id -> speed
signal_memory = set()     # ids that already violated signal
collision_counter = {}    # (id1, id2) -> frames with crash‑like behavior
collision_age = {}        # (id1, id2) -> frames since last overlap

crash_flash_counter = 0
frame_idx = 0


#draw alert

def draw_alerts(frame, events):
    """Draw textual alerts except the big ACCIDENT banner."""
    y = 80
    mapping = {
        "jaywalking":       ("JAYWALKING DETECTED", (0, 165, 255)),
        "speeding":         ("SPEEDING DETECTED", (0, 0, 255)),
        "signal_violation": ("SIGNAL VIOLATION", (0, 0, 255)),
        "drowsy":           ("DRIVER DROWSY", (255, 0, 255)),
        "phone":            ("PHONE DISTRACTION", (255, 255, 0)),
    }

    for e in sorted(events):
        if e == "accident":
            continue
        if e in mapping:
            text, color = mapping[e]
            cv2.putText(
                frame, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3
            )
            y += 35


#main loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    events = set()
    crash_boxes = []
    crash_active = False

    #draw roi
    for roi, c in [
        (ROAD_ROI, (255, 0, 0)),
        (ZEBRA_ROI, (0, 255, 0)),
        (STOP_LINE_ROI, (0, 0, 255)),
    ]:
        pts = np.array(list(roi.exterior.coords), np.int32)
        cv2.polylines(frame, [pts], True, c, 2)

    #yolo detection
    results = model(frame, verbose=False)

    persons, phones, vehicle_dets = [], [], []

    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls = model.names[int(b.cls[0])]
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            if cls == "person":
                persons.append([x1, y1, x2, y2])
            elif cls == "cell phone":
                phones.append([x1, y1, x2, y2])
            elif cls in VEHICLES:
                vehicle_dets.append(
                    [x1, y1, x2, y2, float(b.conf[0]), cls]
                )

    #jay walking [preframe - temporal???]
    person_prev = {}
    for i, p in enumerate(persons):
        cx, cy = box_center(p)
        pt = Point(cx, cy)

        moved = (
            i in person_prev
            and np.hypot(cx - person_prev[i][0], cy - person_prev[i][1])
            > PERSON_MOVE_THRESH
        )

        if moved and ROAD_ROI.contains(pt) and not ZEBRA_ROI.contains(pt):
            events.add("jaywalking")

        person_prev[i] = (cx, cy)

    #phone distraction
    for p in persons:
        for ph in phones:
            if iou(p, ph) > 0.05:
                events.add("phone")

    #tracking + speed
    tracks = vehicle_tracker.update(vehicle_dets, frame)
    track_data = {}  # track_id -> {box, speed}

    for t in tracks:
        if not t.is_confirmed():
            continue

        tid = t.track_id
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # crude speed estimate in “km/h‑like” units
        speed = SPEED_LIMIT
        if tid in vehicle_pos:
            px, py = vehicle_pos[tid]
            dist = np.hypot(cx - px, cy - py)
            speed = dist * 0.05 * FPS
        speed_memory[tid] = speed
        vehicle_pos[tid] = (cx, cy)

        if speed > SPEED_LIMIT:
            events.add("speeding")

        # signal violation
        if (
            SIGNAL_STATE == "RED"
            and STOP_LINE_ROI.contains(Point(cx, cy))
            and tid not in signal_memory
        ):
            events.add("signal_violation")
            signal_memory.add(tid)

        track_data[tid] = {"box": [x1, y1, x2, y2], "speed": speed}

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame, f"ID {tid}", (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )


    # ACCIDENT DETECTION (REVISED)
    # condition: vehicles overlap + at least one has
    # meaningful impact speed, followed by both slowing down
  
    ids = list(track_data.keys())

    # decay collision counters when no longer overlapping
    for key in list(collision_age.keys()):
        collision_age[key] += 1
        if collision_age[key] > COLLISION_DECAY:
            collision_age.pop(key, None)
            collision_counter.pop(key, None)

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            b1, b2 = track_data[id1]["box"], track_data[id2]["box"]

            ov = iou(b1, b2)
            key = tuple(sorted((id1, id2)))

            if ov > 0.15:
                collision_age[key] = 0

                s1 = speed_memory.get(id1, 0.0)
                s2 = speed_memory.get(id2, 0.0)

                # impact: at least one vehicle moving fast enough
                impact_like = max(s1, s2) > MIN_CRASH_SPEED

                if impact_like:
                    collision_counter[key] = collision_counter.get(key, 0) + 1

                    if collision_counter[key] >= ACCIDENT_FRAME_THRESH:
                        crash_active = True
                        events = {"accident"}
                        crash_boxes.extend([b1, b2])
            else:
                # no overlap in this frame -> age will decay
                pass

    #ui

    if crash_active:
        crash_flash_counter += 1
        if (crash_flash_counter // FLASH_PERIOD) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(
                overlay, (0, 0), frame.shape[1::-1], (0, 0, 255), -1
            )
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        for x1, y1, x2, y2 in crash_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        cv2.putText(
            frame, "ACCIDENT DETECTED",
            (frame.shape[1] // 2 - 220, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )
    else:
        crash_flash_counter = 0

    #drowsiness

    frame, alarm = drowsy_detector.process(
        frame, thresholds=DROWSINESS_THRESHOLDS
    )
    if alarm and not crash_active:
        events.add("drowsy")

    #risk + alert

    risk_score = compute_risk(events)
    cv2.putText(
        frame, f"Risk: {risk_score:.2f}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
    )

    draw_alerts(frame, events)
    cv2.imshow("AI VISION – CCTV SAFETY", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
