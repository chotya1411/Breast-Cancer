# Breast Cancer Malignancy Prediction

A machine learning project that trains several scikit-learn classifiers on the Wisconsin Diagnostic Breast Cancer dataset and provides a Streamlit dashboard for exploring the data, comparing models, and making interactive predictions.

## Features

- Loads the Wisconsin Diagnostic Breast Cancer dataset from scikit-learn.
- Splits and standardizes the data using a reproducible train/test split.
- Trains four models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
- Evaluates models using accuracy, precision, recall, F1 score, and ROC AUC.
- Selects and saves the best model based on ROC AUC.
- Generates ROC curve, confusion matrix, and feature importance plots.
- Provides a Streamlit interface with dataset exploration, model comparison, and an interactive predictor.

## Requirements

- Python 3.9 or newer
- Windows PowerShell, Command Prompt, or another terminal

The dataset is included with scikit-learn, so no separate dataset download is required.

## Installation

Open a terminal in the project directory:

```powershell
cd D:\AI-ML
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, run this once for the current user, then activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn joblib matplotlib streamlit
```

## Train the Models

Run the training pipeline from the project root:

```powershell
python main.py
```

This command loads the data, trains and evaluates all models, saves the best model and scaler, and creates the visualizations in `artifacts/`.

The generated files are:

- `artifacts/best_model.joblib`: the model with the highest ROC AUC
- `artifacts/scaler.joblib`: the fitted `StandardScaler`
- `artifacts/metrics.json`: evaluation metrics for all models
- `artifacts/roc_curves.png`: ROC curves for all models
- `artifacts/confusion_matrix.png`: normalized confusion matrix for the best model
- `artifacts/feature_importance.png`: top feature importances

Run the training command again whenever you want to regenerate these artifacts.

## Run the Streamlit Dashboard

After training, start the web app:

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in a browser. The dashboard contains three pages:

1. **Dataset Explorer** - view dataset statistics, samples, and class distribution.
2. **Model Comparison** - compare evaluation metrics and view generated plots.
3. **Interactive Predictor** - adjust the 30 input features and predict benign or malignant classification.

Stop the dashboard with `Ctrl+C` in the terminal.

## Run Everything with One Command

From the project root, use this PowerShell command to train the models and then start the dashboard:

```powershell
python main.py; if ($LASTEXITCODE -eq 0) { python -m streamlit run app.py }
```

## Project Structure

```text
.
├── app.py                    # Streamlit dashboard
├── config.py                 # Paths, feature names, and project settings
├── main.py                   # End-to-end training pipeline
├── artifacts/                # Saved models, metrics, and plots
└── src/
    ├── data/
    │   ├── dataset.py        # Dataset loading helpers
    │   └── preprocess.py     # Train/test split and scaling
    ├── models/
    │   └── train.py          # Model training and evaluation
    └── visualization/
        └── plots.py          # Evaluation plot generation
```

## Notes

- The target labels come from scikit-learn: `0` is malignant and `1` is benign.
- The predictor uses the saved scaler and best model from `artifacts/`.
- This project is for educational and demonstration purposes and is not a medical diagnostic tool.
