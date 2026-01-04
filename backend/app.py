import random
from constants import SEVERITY_ARRAY
from utils import convert_string_to_array
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request
import os
from ml_inference import analyze_video
import uuid
from config import UPLOAD_FOLDER_PATH
from config import db_instance
from models.incident_history import IncidentHistory


app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DB_URI")

db_instance.init_app(app)

with app.app_context():
    db_instance.create_all()

@app.post("/detection")
def detection():


    address = request.form.get('address')
    if not address:
        return jsonify({"error": "No address provided"}), 400  
    

    location = convert_string_to_array(request.form.get('location'))
    if not location:
        return jsonify({"error": "Invalid location"}), 400
  

    if "file" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    # print(request.form)
    file = request.files["file"]
    request_id = uuid.uuid4()
    video_path = os.path.join(UPLOAD_FOLDER_PATH, f"{request_id}.mp4")
    file.save(video_path)

    result = analyze_video(video_path)

    severity = random.choice(SEVERITY_ARRAY)
    result['severity'] = severity

    try:
        new_incident = IncidentHistory(incident_id = request_id, address = address, status = 'confirmed' if result['accident'] else 'negative', confidence_score = result["confidence"], severity = severity, coordinates = location)
        db_instance.session.add(new_incident)
        db_instance.session.commit()


        return jsonify({"status":True, "msg":"Incident detection succeeded", "result":result}),201

    except Exception as e: 
        print(e)
        return jsonify({"status":False, "msg":"Internal server error"}),500
    
@app.get("/incidents")
def get_incidents():
    try:
        incidents = IncidentHistory.query.all()
        return jsonify({"status":True, "msg":"Incidents fetched", "incidents":[inc.to_dict() for inc in incidents]}),200
    except Exception as e: 
        print(e)
        return jsonify({"status":False, "msg":"Internal server error"}),500

@app.get("/")
def get_route():
    return jsonify({"msg" : "Backend Running Successfully"})


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER_PATH,exist_ok=True)
    print("Flask server starting on http://127.0.0.1:5000")

    app.run(host="0.0.0.0", port=5000, debug=True)