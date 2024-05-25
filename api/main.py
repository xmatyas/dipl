import os
import uuid
from flask import request, jsonify, abort
from flask_restful import Resource
from .app import flask_app, flask_api, celery_app

# Local import
from . import tasks                   # Import the Celery task from folder 'api' and file 'tasks' (behaves like a module)

# Set the UPLOAD_FOLDER and ALLOWED_EXTENSIONS environment variables
UPLOAD_FOLDER = flask_app.config['FLASK']['upload_folder']
ALLOWED_EXTENSIONS = flask_app.config['FLASK']['allowed_extensions']

###############################################################
#               POST /upload util functions                   #
###############################################################
def allowed_file(filename, extensions):
    if get_file_extension(filename, dot = False) in extensions:
        return True
    return False

def get_file_extension(filename, dot : bool = True):
    # Get the file extension with the dot
    if dot:
        return os.path.splitext(filename)[1].lower()
    # Get the file extension without the dot
    return os.path.splitext(filename)[1][1:].lower()

def validate_post_request(request, extensions):
    # Validation from simplest to most complex checks
    if 'file' not in request.files:
        abort(400, 'File part not found in request')
    file = request.files['file']
    if file is None:
        abort(400, 'File part not found in request')
    filename = file.filename
    if filename == '':
        abort(400, 'Empty file name in request')
    if not allowed_file(filename, extensions):
        abort(415, 'Unsupported Media Type')
    return file, filename

def save_file(file, filename, upload_folder):
    extension = get_file_extension(filename)
    filename = str(uuid.uuid4()) + extension
    if os.path.exists(upload_folder) is False:
        os.makedirs(upload_folder)
    file.save(os.path.join(upload_folder, filename))
    path = os.path.join(upload_folder, filename)
    return path
    
###############################################################
#               GET /result util functions                    #
###############################################################
def validate_uuid(id):
    try:
        uuid.UUID(id)
    except ValueError:
        abort(400, 'Invalid UUID format')
        

class UploadFile(Resource):
    def post(self):
        # Check if the post request has the file part
        file, filename = validate_post_request(request, ALLOWED_EXTENSIONS)
        abs_path = os.path.abspath(UPLOAD_FOLDER)
        print(f'File uploaded: {filename} to {abs_path}')
        # Save file to UPLOAD_FOLDER and return success message
        save_path = save_file(file, filename, abs_path)
        
        # Start celery task asynchronously, catch unknown exceptions
        try:
            task = tasks.process_img.delay(save_path)
        except Exception as e:
            os.remove(save_path)
            abort(500, 'Failed to start Celery task')
        
        data = {
            'task_id': task.id, 
            'message': 'File uploaded and processing started',
        }
        return data, 202

class GetResult(Resource):
    def get(self, id):
        # Validate format uuid
        validate_uuid(id)
        # Get the result of the task
        task = tasks.process_img.AsyncResult(id)
        result = task.get(timeout=10)
        return jsonify({'result': result})

# Add the routes to the Flask-RESTful API
flask_api.add_resource(UploadFile, '/upload')
flask_api.add_resource(GetResult, '/result/<string:id>')

if __name__ == '__main__':
    # Run the Flask app
    flask_app.run()
