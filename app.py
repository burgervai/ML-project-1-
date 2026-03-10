from flask import Flask, request, jsonify, render_template
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from sklearn.metrics import r2_score
from src.utilitis import save_object, evaluate_models, load_object  # Added load_object assuming it's available in utilitis
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np  
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import webbrowser
import time
import threading  


app = Flask(__name__)

@app.route('/')
def index():
    # serve the prediction form directly as the homepage
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        # display the form if accessed via GET
        return render_template('home.html')
    else:
        try:
            # Safely parse numeric fields
            reading_score = float(request.form.get('reading_score')) if request.form.get('reading_score') else None
            writing_score = float(request.form.get('writing_score')) if request.form.get('writing_score') else None
            
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=reading_score,
                writing_score=writing_score
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=results[0])
        except Exception as e:
            return render_template('home.html', results=f"Error: {str(e)}")
    
if __name__ == "__main__":
    def open_browser():
        time.sleep(1.5)  # Wait a bit for the server to start
        webbrowser.open("http://127.0.0.1:5000/")
    
    threading.Thread(target=open_browser).start()
    app.run(host="0.0.0.0", debug=True)
    

    