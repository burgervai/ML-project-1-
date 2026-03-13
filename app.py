# app.py
import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import traceback
import pickle

# -----------------------------
# Flask app initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load the trained model
# -----------------------------
MODEL_PATH = os.path.join("artifacts", "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print(f"Warning: Model file not found at {MODEL_PATH}")

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Make sure index.html exists in templates/

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", prediction_text="Error: Model not found. Please check artifacts/model.pkl")
    
    try:
        # Collect form data safely
        data = {
            "gender": request.form.get("gender"),
            "race_ethnicity": request.form.get("race_ethnicity"),
            "parental_level_of_education": request.form.get("parental_level_of_education"),
            "lunch": request.form.get("lunch"),
            "test_preparation_course": request.form.get("test_preparation_course"),
            "writing_score": float(request.form.get("writing_score", 0)),
            "reading_score": float(request.form.get("reading_score", 0))
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        return render_template("index.html", prediction_text=f"Predicted Exam Score: {prediction:.2f}")
    
    except Exception as e:
        print(traceback.format_exc())
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render dynamically assigns this
    app.run(host="0.0.0.0", port=port)
