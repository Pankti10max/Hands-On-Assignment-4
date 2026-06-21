"""
Fraud Detection using PyOD AutoEncoder
"""

import pandas as pd
import numpy as np
from pyod.models.auto_encoder import AutoEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Initialize AutoEncoder
model = AutoEncoder(
    hidden_neuron_list=[64, 32, 32, 64],
    epochs=20,
    batch_size=32,
    contamination=0.0017,
    verbose=1
)

# Train model
model.fit(X_train)

# Predict anomalies
y_pred = model.predict(X_test)

# Results
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))
