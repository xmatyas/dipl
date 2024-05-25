from flask import Flask
from flask_restful import Api
from celery import Celery
from . import celery_init
from . import config_loader
import os

# Create Flask app
flask_app = Flask(__name__)
# Create Flask-RESTful API
flask_api = Api(flask_app)

# Load configuration from config.yaml
config_file_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config_loader.load_config(flask_app, config_file_path)

# Initialize Celery with Flask app context
celery_app = celery_init.celery_init_app(flask_app)