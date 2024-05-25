"""
Letter recognition module
"""
from os import environ
from . import model_usage

# Set the environment variables
environ['MODEL_FOLDER'] = 'models'
environ['MODEL_NAME'] = 'emnist_model'
# Set the version of the package
version = '0.0.1'