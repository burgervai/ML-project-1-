## End to End machine learning project

This repository contains a complete end-to-end machine learning pipeline for predicting student math exam scores based on demographic and academic features. The project includes data ingestion, transformation, model training (with ensemble learning), and a Flask web application for making live predictions.

## Project Structure

```
ml project 1/
├── app.py                   # Flask application for web UI
├── requirements.txt         # Python dependencies
├── setup.py
├── artifacts/               # Generated data, models, and preprocessors
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl            # Best model (could be ensemble)
│   └── preprocessor.pkl     # Preprocessing pipeline
├── dataset/                 # Original datasets and notebooks
│   └── stud.csv
├── logs/                    # Training logs
├── src/                     # Python source code
│   ├── components/          # Individual pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── __init__.py
│   ├── pipeline/            # Orchestration scripts
│   │   ├── predict_pipeline.py
│   │   ├── train_pipeline.py
│   │   ├── __init__.py
│   │   └── ...
│   ├── exception.py
│   ├── logger.py
│   └── utilitis.py
├── templates/               # HTML templates for Flask UI
│   ├── home.html
│   └── index.html
├── tests/                   # (optional) unit tests
└── README.md
```

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/burgervai/ML-project-1.git
   cd "ml project 1"
   ```

2. **Create a virtual environment and install dependencies**
   ```powershell
   python -m venv venv
   & "venv\\Scripts\\Activate.ps1"  # Windows PowerShell
   pip install -r requirements.txt
   ```

3. **Train the models**
   The training script ingests the raw CSV, preprocesses the data, trains multiple regressors, evaluates them, and saves the best model (or a voting ensemble) to `artifacts/model.pkl`.
   ```powershell
   & "venv\\Scripts\\Activate.ps1"
   python -m src.pipeline.train_pipeline
   ```

4. **Run the web application**
   After training, start the Flask app to make predictions via a browser-based form:
   ```powershell
   & "venv\\Scripts\\Activate.ps1"
   python app.py
   ```
   The app will open at `http://127.0.0.1:5000/`.

## Features

- **Data Ingestion**: reads `dataset/stud.csv`, saves raw/train/test splits.
- **Data Transformation**: handles missing values, encodes categoricals, scales features.
- **Model Training**: evaluates a suite of regressors and optionally a voting ensemble; selects best model based on R².
- **Prediction API**: Flask form collects inputs and returns predicted math score.
- **Modular design**: components are reusable and easy to extend.

## Notes

- The project uses `np.nan` for missing values to ensure compatibility with scikit-learn.
- The ensemble is a `VotingRegressor` trained on all candidate models; it will be chosen if it outperforms any single model.
- Logging is configured in `src/logger.py` and outputs to `logs/`.

## Future Improvements

- Add unit and integration tests.
- Parameterize training pipeline for hyperparameter tuning.
- Deploy the Flask app using Docker or a cloud service.

---

Feel free to explore, modify the models, or plug in new datasets. Happy coding! 🎓
