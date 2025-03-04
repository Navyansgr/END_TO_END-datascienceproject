import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset (you can also save it in the data/ folder and load locally)
df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")

# Handle missing values (here we simply drop them)
df.dropna(inplace=True)

# Convert categorical data to numerical
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Split into features and target
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler and preprocessed dataset for later use
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), "models/dataset.pkl")

print("Data preprocessing completed!")