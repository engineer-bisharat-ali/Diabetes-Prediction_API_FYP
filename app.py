from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
package = joblib.load("diabetes_model.pkl")
model = package["model"]
scaler = package["scaler"]
gender_map = package["gender_map"]
smoke_map = package["smoke_map"]
feature_order = package["feature_order"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Diabetes Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = []
        for feature in feature_order:
            val = data.get(feature)
            if feature == "gender":
                val = gender_map.get(val, 0)
            elif feature == "smoking_history":
                val = smoke_map.get(val, 0)
            input_data.append(val)

        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        result = "Diabetes" if prediction == 1 else "No Diabetes"

        return jsonify({
            "prediction": result,
            "confidence_score": f"{round(probability * 100, 2)}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)