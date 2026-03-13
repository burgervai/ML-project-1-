# app.py
import os
from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

MODEL_PATH = "artifacts/model.pkl"
model = None

def load_model():
    global model
    if model is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Model file not found at {MODEL_PATH}")
    return model

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/form")
def form_page():
    return render_template("home.html", results="")

@app.route("/predict", methods=["POST"])
def predict():
    model_instance = load_model()
    if model_instance is None:
        return "Error: Model not loaded. Check artifacts/model.pkl", 500

    try:
        data = {
            "gender": request.form.get("gender"),
            "race_ethnicity": request.form.get("ethnicity"),
            "parental_level_of_education": request.form.get("parental_level_of_education"),
            "lunch": request.form.get("lunch"),
            "test_preparation_course": request.form.get("test_preparation_course"),
            "writing_score": float(request.form.get("writing_score", 0)),
            "reading_score": float(request.form.get("reading_score", 0))
        }
        df = pd.DataFrame([data])
        prediction = model_instance.predict(df)[0]
        return render_template("home.html", results=f"{prediction:.2f}")
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
