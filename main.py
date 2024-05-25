import api
import recog
import ocr
from api.main import flask_app
from api.main import celery_app

if __name__ == '__main__':
    flask_app.run(debug=True, host='127.0.0.1', port=5000)
    print("All modules prepared. Ready to run.")