from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request
import os
from ml_inference import analyze_video
import uuid
from config import UPLOAD_FOLDER_PATH



app = Flask(__name__)



@app.post("/detection")
def detection():
    if "file" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["file"]
    request_id = uuid.uuid4()
    video_path = os.path.join(UPLOAD_FOLDER_PATH, f"{request_id}.mp4")
    file.save(video_path)

    result = analyze_video(video_path)

    print("Prediction result:", result)

    return jsonify(result)

@app.get("/")
def get_route():
    return jsonify({"msg" : "Backend Running Successfully"})


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER_PATH,exist_ok=True)
    print("Flask server starting on http://127.0.0.1:5000")

    app.run(host="0.0.0.0", port=5000, debug=True)