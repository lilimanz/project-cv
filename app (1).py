from flask import Flask, jsonify
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

DATA_URL = 'https://api.coincap.io/v2/assets/bitcoin/history'
MODEL_FILE = 'random_forest_model.pkl'
SCALER_FILE = 'scaler.pkl'

def fetch_data():
    END_DATE = datetime.utcnow()
    START_DATE = END_DATE - timedelta(days=15)
    response = requests.get(DATA_URL, params={
        'interval': 'h1',
        'start': int(START_DATE.timestamp() * 1000),
        'end': int(END_DATE.timestamp() * 1000)
    })
    data = response.json()['data']
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['time']/1000, unit='s')
    df.set_index('date', inplace=True)
    df['priceUsd'] = df['priceUsd'].astype(float)
    df['circulatingSupply'] = df['circulatingSupply'].astype(float)
    return df[['priceUsd', 'circulatingSupply']]

def preprocess_data(data):
    data['priceUsd_lag1'] = data['priceUsd'].shift(1)
    data['priceUsd_lag2'] = data['priceUsd'].shift(2)
    data['priceUsd_lag3'] = data['priceUsd'].shift(3)
    data.dropna(inplace=True)
    X = data[['priceUsd_lag1', 'priceUsd_lag2', 'priceUsd_lag3', 'circulatingSupply']]
    return X

@app.route('/predict', methods=['GET'])
def predict():
    data = fetch_data()
    X = preprocess_data(data)
    
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled[-1].reshape(1, -1))
    
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
