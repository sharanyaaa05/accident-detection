import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from detection.main import detect_accident

def analyze_video(video_path):
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    print(cap.isOpened())

    while cap.isOpened():
        ret, frame = cap.read()
        print(ret)
        print(frame)
        if not ret:
            print("1234")
            break

        accident, confidence = detect_accident(frame)
        print(f"Accident is {accident} and conf is {confidence}")

        if accident:
            cap.release()
            return {
                "accident": True,
                "confidence": confidence
            }

    cap.release()
    return {
        "accident": False,
        "confidence": 0.0
    }
