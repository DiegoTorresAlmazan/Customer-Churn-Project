import pandas as pd
import os

def preprocess_data():
    """
    This script loads the raw Telco churn data, cleans it, performs
    feature engineering, and saves the final processed data to be used
    in the model training phase.
    """
    # --- Pathing Setup ---
    # Get the absolute path to the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get the project root
    project_root = os.path.dirname(script_dir)
    
    # Define paths relative to the project root
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'Telco-Customer-Churn.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'churn_processed_data.csv')
    processed_dir = os.path.join(project_root, 'data', 'processed')

    # Create directory for processed data if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Load raw data
    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)

    # --- Data Cleaning ---
    print("Cleaning data...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.drop(columns=['customerID'], inplace=True)
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_total_charges, inplace=True)
    
    # --- Feature Engineering ---
    print("Performing feature engineering...")
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    df_processed = df.copy()
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # Create and encode tenure bins
    labels = ['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5+yr']
    df_processed['tenure_bins'] = pd.cut(df_processed['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], labels=labels, right=False)
    df_processed = pd.get_dummies(df_processed, columns=['tenure_bins'], drop_first=True)

    # --- Save Processed Data ---
    df_processed.to_csv(processed_data_path, index=False)
    
    print(f"Preprocessing complete. Processed data saved to '{processed_data_path}'")

if __name__ == '__main__':
    preprocess_data()
