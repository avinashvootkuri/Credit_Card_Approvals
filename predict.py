import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

model_file = 'model_eta=0.01.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('creditcard-approval')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    D = xgb.DMatrix(X)
    y_pred = model.predict(D)
    Approved = y_pred >= 0.6

    result = {
        'ApprovalChances': float(y_pred),
        'Approved': bool(Approved)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9695)


# print('input', customer)
# print('Credit Card Approved/Denied', y_pred
#     #   (y_pred >= 0.6).astype(int)
#       )