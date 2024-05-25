from celery import Celery
from flask import Flask #Purely for the app context

# Configure Celery
def celery_init_app(flask_app: Flask) -> Celery:
    celery_app = Celery(flask_app.name)
    # Update the Celery configuration with the Flask configuration, loaded from config.yaml
    celery_app.conf.update(flask_app.config['CELERY'])
    celery_app.set_default()
    flask_app.extensions["celery"] = celery_app
    return celery_app