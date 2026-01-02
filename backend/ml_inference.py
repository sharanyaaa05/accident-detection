import cv2
from detection.main import detect_accident

def analyze_video(video_path):
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        accident, confidence = detect_accident(frame)

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
