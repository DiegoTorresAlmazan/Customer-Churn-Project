# Customer Churn Prediction for B2B SaaS

This repository contains an end-to-end machine learning project to predict customer churn for a fictional B2B SaaS company, "SynthSaaS". The project is structured to emulate a real-world production environment, from data preprocessing to model deployment via a REST API and an interactive dashboard.

## 1. Business Problem

SynthSaaS is experiencing a high customer churn rate, which negatively impacts its Monthly Recurring Revenue (MRR) and long-term growth. This project aims to solve this by building a model that can proactively identify customers who are at high risk of churning. By identifying these customers, the business can launch targeted retention campaigns (e.g., discounts, additional support, feature training) to reduce churn and increase Customer Lifetime Value (CLV).

## 2. Technical Pipeline

The project is broken down into a series of modular, executable scripts and notebooks:

-   **Data Exploration & Interpretation (`/notebooks`)**: Jupyter notebooks for initial EDA, feature analysis, and model interpretation using SHAP.
-   **Data Preprocessing (`/src/preprocess.py`)**: A script that loads the raw data, cleans it, and performs feature engineering. It saves the processed, model-ready data.
-   **Model Training (`/src/train.py`)**: A script that loads the processed data, handles class imbalance (using SMOTE), trains multiple classifiers (Logistic Regression, Random Forest, XGBoost), evaluates them, and saves the best-performing model.
-   **Prediction API (`/api/main.py`)**: A FastAPI application that serves the trained model. It exposes a `/predict` endpoint that takes new customer data and returns a real-time churn prediction.
-   **Interactive Dashboard (`/dashboard/main.py`)**: A Streamlit web application that provides a user-friendly interface for making predictions via the API and displays key business insights from the model.

## 3. How to Run This Project

Follow these steps to run the entire pipeline from your local machine.

### Prerequisites

-   Python 3.8+
-   Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/DiegoTorresAlmazan/Customer-Churn-Project
cd <repository-name>
```

### Step 2: Install Dependencies

Install all required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 3: Run the Data Pipeline

First, run the preprocessing script to generate the model-ready data. Then, run the training script to train the models and save the best one.

```bash
# Generate the processed data
python src/preprocess.py

# Run the training pipeline
python src/train.py
```

### Step 4: Run the API and Dashboard

You will need two separate terminals for this step.

**In your first terminal**, start the FastAPI server:

```bash
uvicorn api.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

**In your second terminal**, start the Streamlit dashboard:

```bash
streamlit run dashboard/main.py
```
The dashboard will open in your browser, available at `http://localhost:8501`.

## 4. Key Findings & Model Performance

-   **Best Model**: The final model selected was **Logistic Regression**, which achieved a **ROC-AUC of 0.82** on the test set.
-   **Top Churn Drivers**:
    1.  **Contract Type**: Customers on `Month-to-month` contracts are at the highest risk.
    2.  **Customer Tenure**: New customers (low tenure) are far more likely to churn.
    3.  **Internet Service**: Customers with `Fiber optic` internet service show a higher churn rate.
-   **Actionable Insight**: The most impactful retention strategy would be to create targeted incentives for new, month-to-month customers to upgrade to one or two-year contracts.
