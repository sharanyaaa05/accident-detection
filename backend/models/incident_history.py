from utils import get_timestamp_in_millis
from config import db_instance



class IncidentHistory(db_instance.Model):
    __tablename__ = "incident_history"

    incident_id = db_instance.Column(db_instance.String, primary_key=True)
    timestamp = db_instance.Column(db_instance.BigInteger, default=get_timestamp_in_millis)
    address = db_instance.Column(db_instance.String, nullable=False)
    status = db_instance.Column(db_instance.String, nullable=False)
    confidence_score = db_instance.Column(db_instance.Float, nullable=False)
    severity = db_instance.Column(db_instance.String, nullable=False)
    coordinates = db_instance.Column(db_instance.ARRAY(db_instance.Float))

    def to_dict(self):
        return {
            "incident_id" : self.incident_id,
            "timestamp" : self.timestamp,
            "address" : self.address,
            "status" : self.status,
            "confidence_score" : self.confidence_score,
            "severity" : self.severity,
            "coordinates" : self.coordinates
        }
