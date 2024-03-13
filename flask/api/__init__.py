from flask import Flask
from flask_restful import Api
from api.config_loader import load_config
from api import celery_init
import os

# Create Flask app
flask_app = Flask(__name__)
# Create Flask-RESTful API  
flask_api = Api(flask_app)

# Load configuration from config.yaml
config_file_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
load_config(flask_app, config_file_path)

# Initialize Celery with Flask app context
celery_app = celery_init.celery_init_app(flask_app)

# Import needed for flask_api routes to be added
from api import main
