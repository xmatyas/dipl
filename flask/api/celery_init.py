from celery import Celery, Task
from flask import Flask #Purely for the app context

# Configure Celery
def celery_init_app(flask_app: Flask) -> Celery:
    celery_app = Celery(flask_app.name, 
                        broker=flask_app.config['CELERY']['BROKER_URL'],
                        #result_backend=flask_app.config['CELERY']['RESULT_BACKEND'],
                        backend=flask_app.config['CELERY']['RESULT_BACKEND'],
                        broker_connection_retry_on_startup=flask_app.config['CELERY']['BROKER_CONNECTION_RETRY_ON_STARTUP']
                        )
    celery_app.set_default()
    flask_app.extensions["celery"] = celery_app
    return celery_app