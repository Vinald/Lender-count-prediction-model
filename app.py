import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

model_file_path = 'model.pkl'

if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"Model file '{model_file_path}' was not found.")

with open(model_file_path, 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        loan_amount = int(request.form['loanAmount'])
        repayment_term = int(request.form['repaymentTerm'])

        features = np.array([[loan_amount, repayment_term]])
        predictions = model.predict(features)
        prediction_result = f'Predicted Status: {predictions[0]}'

        return render_template('index.html', 
                               prediction_result=prediction_result)

    except Exception as e:
        return render_template('index.html', 
                               prediction_result=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
