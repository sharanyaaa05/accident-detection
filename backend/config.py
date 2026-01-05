import os
from flask_sqlalchemy import SQLAlchemy

UPLOAD_FOLDER_PATH=os.getenv("UPLOAD_FOLDER_PATH")
db_instance=SQLAlchemy()