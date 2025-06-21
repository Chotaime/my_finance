from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

# Create the Flask app
app = Flask(__name__)
CORS(app)  # Enables CORS for all domains by default


@app.route('/')
def home():
    return 'Server is working!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

# Only needed for local testing
if __name__ == '__main__':
    app.run(debug=True)

