from flask import Flask, jsonify, request
# from flask_sqlalchemy import SQLAlchemy

import os
from ml_inference import analyze_video




app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost:5432/DB'

# db = SQLAlchemy(app)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/analyze")
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["file"]
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    print("Video received:", file.filename)

    result = analyze_video("video.mp4")

    print("Prediction result:", result)

    return jsonify(result)

@app.get("/")
def get_route():
    return jsonify({"msg" : "Backend Running Successfully"})


if __name__ == '__main__':
    print("🚀 Flask server starting on http://127.0.0.1:5000")

    app.run(host="0.0.0.0", port=5000, debug=True)