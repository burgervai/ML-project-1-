import os
from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# -----------------------------
# Load model and preprocessor lazily
# -----------------------------
MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

model = None
preprocessor = None

def load_model_and_preprocessor():
    global model, preprocessor
    if model is None or preprocessor is None:
        try:
            with open(PREPROCESSOR_PATH, "rb") as f:
                preprocessor = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Preprocessor not found at {PREPROCESSOR_PATH}")
            preprocessor = None

        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Model not found at {MODEL_PATH}")
            model = None
    return model, preprocessor

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home_page():
    """Landing page"""
    return render_template("index.html")  # Landing page

@app.route("/form")
def form_page():
    """Prediction form page"""
    return render_template("home.html", results="")  # Form page

@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and return prediction"""
    model_instance, preprocessor_instance = load_model_and_preprocessor()

    if model_instance is None or preprocessor_instance is None:
        return "Error: Model or preprocessor not loaded. Check artifacts folder.", 500

    try:
        # Get form data
        data = {
            "gender": request.form.get("gender"),
            "race_ethnicity": request.form.get("ethnicity"),
            "parental_level_of_education": request.form.get("parental_level_of_education"),
            "lunch": request.form.get("lunch"),
            "test_preparation_course": request.form.get("test_preparation_course"),
            "writing_score": float(request.form.get("writing_score", 0)),
            "reading_score": float(request.form.get("reading_score", 0))
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Apply preprocessing
        X_input = preprocessor_instance.transform(df)

        # Predict
        prediction = model_instance.predict(X_input)[0]

        # Render form with prediction
        return render_template("home.html", results=f"{prediction:.2f}")

    except Exception as e:
        return f"Error: {e}", 500

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
