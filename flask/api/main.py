import os
import yaml
from flask import request, jsonify, abort, Flask
from flask_restful import Resource
# Local import
from api import flask_app, flask_api    # Import the Flask app and the Flask-RESTful API from the __init__.py file in the 'api' folder
from api import tasks                   # Import the Celery task from folder 'api' and file 'tasks' (behaves like a module)

ALLOWED_EXTENSIONS = flask_app.config['FLASK']['ALLOWED_EXTENSIONS']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class HelloWorldResource(Resource):
    def get(self):
        return {'message': 'Hello, World!'}

class UploadFile(Resource):
    def post(self):
        # Check if the post request has the file part
        if 'file' not in request.files:
            abort(400, 'No file part')
        file = request.files['file']
        if file.filename == '':
            abort(400, 'No selected file')
        
        # Check if filetype is in the ALOWED_EXTENSIONS set
        if not allowed_file(file.filename):
            abort(400, 'Invalid file type')
        
        # Save file to UPLOAD_FOLDER and return success message
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(flask_app.config['FLASK']['UPLOAD_FOLDER'], filename))
            
            # Start celery task asynchronously
            task = tasks.process_pdf.delay(os.path.join(flask_app.config['FLASK']['UPLOAD_FOLDER'], filename))            

            return {'result_id': task.id, }
        else:
            abort(400, 'Invalid file type')

class GetResult(Resource):
    def get(self, id):
        task = tasks.process_pdf.AsyncResult(id)
        print(task)
        # if task.state == 'PENDING' :
        #     response = {
        #         'state': task.state,
        #         #'status': task.status
        #     }
        # elif task.state != 'FAILURE' :
        #     response = {
        #         'state': task.state,
        #         #'status': task.status,
        #     }
        #     if 'result' in task.info:
        #         response['result'] = task.info['result']
        # else:
        #     response = {
        #         'state': task.state,
        #         'status': str(task.info),
        #     }
        # return jsonify(response)
        result = None

    # Wait for the result, adjust the timeout as needed
        try:
            result = task.get(timeout=10)
        except TimeoutError:
            result = "Task is still running, check back later."

        return jsonify({'result': result})

flask_api.add_resource(UploadFile, '/upload')
flask_api.add_resource(GetResult, '/result/<string:id>')
flask_api.add_resource(HelloWorldResource, '/hello')