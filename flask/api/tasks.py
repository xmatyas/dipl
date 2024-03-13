import random
from api import celery_app as celery
from api import flask_app
from celery import states, Task

class FlaskTask(Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)
            
@celery.task(bind=True, base=FlaskTask)
def process_pdf(self, file_path) -> tuple[str, int]:
    #self.update_state(state=states.STARTED)
    # OCR LOGIC
    #self.update_state(state=states.SUCCESS)
    return random.randint(1, 100)