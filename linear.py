import joblib
from flask_cors import CORS
from flask import Flask, request

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Load saved model
reg2 = joblib.load('reg.pkl')

@app.route('/', methods=['GET'])
def home():
    return '<h1>API is running!!</h1>'

@app.route('/pred', methods=['GET'])
def predict():
    x = request.args['x']
    x = float(x)
    prediction = reg2.predict([[x]])
    return {'prediction': int(prediction)}


if __name__ == '__main__':
    app.run(port=8000, debug=True)