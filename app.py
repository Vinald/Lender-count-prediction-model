import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model_path = 'lr.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' was not found.")

with open(model_path, 'rb') as f:
    model = pickle.load(f)
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the values from the form
        loan_amount = int(request.form['loan_amount'])
        repayment_term = int(request.form['repayment_term'])
        
        features = np.array([[loan_amount, repayment_term]])
        predictions = model.predict(features)

        prediction_result = f'Predicted results: {predictions[0]}'

        return render_template('index.html', prediction_text=prediction_result)
    except Exception as e:
        print("Error:", e)
        return render_template('index.html', prediction_text="Error occurred. Please try again.")


if __name__ == '__main__':
    app.run(debug=True)