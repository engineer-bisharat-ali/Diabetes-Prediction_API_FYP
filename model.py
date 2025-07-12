import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Encode categorical features
gender_map = {val: idx for idx, val in enumerate(df["gender"].unique())}
df["gender"] = df["gender"].map(gender_map)

smoke_map = {val: idx for idx, val in enumerate(df["smoking_history"].unique())}
df["smoking_history"] = df["smoking_history"].map(smoke_map)

# Prepare data
X = df.drop("diabetes", axis=1)
y = df["diabetes"]
feature_order = list(X.columns)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 📊 Validation Metrics
print("\n🔹 Validation Metrics:")
print(f"Accuracy     : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision    : {precision_score(y_test, y_pred):.4f}")
print(f"Recall       : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score     : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save
package = {
    "model": model,
    "scaler": scaler,
    "gender_map": gender_map,
    "smoke_map": smoke_map,
    "feature_order": feature_order
}

joblib.dump(package, "diabetes_model.pkl", compress=3)
print("\n✅ Model & preprocessor saved as 'diabetes_model.pkl'")
