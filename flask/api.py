import json
from flask import Flask, request, jsonify, abort
app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api():
    data = [1, 2, 3, 4, 5]
    if not data:
        abort(400, 'No data provided')
    return jsonify(data)

@app.route('/')
def index():
    return jsonify({'name': 'Alice',
                    'email': 'alice@gmail.com'})

if __name__ == '__main__':
    app.run(debug=True)