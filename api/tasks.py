import json
import os
from .app import celery_app as celery
from .app import flask_app
from celery import states, Task
from recog import model_usage
from ocr import preprocess

def safe_remove(file_path):
    print(f'Removing file: {file_path}')
    try:
        os.remove(file_path)
        return True
    except OSError as e:
        print(f'Error removing file: {e}')
        return False

class FlaskTask(Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)
            
@celery.task(bind=True, base=FlaskTask, name='tasks.process_img')
def process_img(self, file_path):
    try:
        try:
            block_segment_images = preprocess.process_image(file_path)
            prediction = model_usage.predict_images(block_segment_images)
        except Exception as e:
            safe_remove(file_path)
            raise self.update_state(
                state=states.FAILURE,
                meta={'exc_type': type(e).__name__, 'exc_message': str(e)}
            )
        result = []
        result.append({'json': json.dumps(prediction)})
        remove_state = safe_remove(file_path)
        if remove_state is False:
            result.append({'exc_type': 'OSError', 'exc_message': 'Error removing file', 'file_deletion': 'FAILED'})
        else:
            result.append({'file_deletion': 'SUCCESS'})
        return result
    except Exception as e:
        raise self.update_state(
            state=states.FAILURE,
            meta={'exc_type': type(e).__name__, 'exc_message': str(e)}
        )