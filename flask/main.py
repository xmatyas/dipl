# Import api module (api folder with __init__.py file) and run the Flask app (flask_app.run() method)
from api import flask_app, flask_api, celery_app

if __name__ == "__main__":
    print(flask_api.app.url_map)
    flask_app.run(debug=True)