import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load preprocessed data saved from preprocess_data.py
X_train_scaled, X_test_scaled, y_train, y_test = joblib.load("models/dataset.pkl")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: {mae}")

# Save model to the models/ folder
joblib.dump(model, "models/house_price_model.pkl")
print("Model saved!")