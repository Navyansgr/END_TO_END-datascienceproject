from fastapi import FastAPI
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to House Price Prediction API"}

@app.post("/predict")
def predict(features: list):
    """
    Expecting a list of numerical features in the same order as the model training data.
    Example: [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ...]
    """
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return {"predicted_price": prediction[0]}