from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API")

# --- Pathing and Model Loading ---
# Construct paths relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')

# Load the trained model, scaler, and training columns
try:
    model = joblib.load(os.path.join(models_dir, 'churn_model.joblib'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    # Load training columns to ensure consistency
    train_df = pd.read_csv(os.path.join(data_dir, 'processed', 'churn_processed_data.csv'))
    training_columns = train_df.drop('Churn', axis=1).columns
except FileNotFoundError as e:
    raise RuntimeError("Model artifacts not found. Please run the training script first.") from e

# Define the input data model using Pydantic
# These are the fields a user would provide for a prediction
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API. Use the /predict endpoint for predictions."}

@app.post("/predict")
def predict_churn(customer: Customer):
    """
    Predict churn for a single customer.
    """
    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([customer.dict()])

    # --- Preprocessing ---
    # One-hot encode categorical features
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Create tenure bins
    labels = ['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5+yr']
    input_df_encoded['tenure_bins'] = pd.cut(input_df_encoded['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], labels=labels, right=False)
    input_df_encoded = pd.get_dummies(input_df_encoded, columns=['tenure_bins'], drop_first=True)

    # Align columns with the training data
    # This adds missing columns (if any) and fills them with 0
    # and removes any columns in the input that weren't in the training data
    input_df_aligned = input_df_encoded.reindex(columns=training_columns, fill_value=0)

    # Scale numerical features using the loaded scaler
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df_aligned[numerical_features] = scaler.transform(input_df_aligned[numerical_features])

    # --- Prediction ---
    prediction = model.predict(input_df_aligned)[0]
    probability = model.predict_proba(input_df_aligned)[0][1] # Probability of churn (class 1)

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": float(probability)
    }

# To run this API:
# 1. Navigate to the project root in your terminal.
# 2. Run the command: uvicorn api.main:app --reload
