import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_training():
    """
    Loads data, preprocesses it, trains multiple models, evaluates them,
    and saves the best-performing model.
    """
    # --- Pathing Setup ---
    # Get the absolute path to the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get the project root
    project_root = os.path.dirname(script_dir)
    
    # Define paths relative to the project root
    data_path = os.path.join(project_root, 'data', 'processed', 'churn_processed_data.csv')
    models_dir = os.path.join(project_root, 'models')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    model_path = os.path.join(models_dir, 'churn_model.joblib')

    # Create directory for models if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Load the processed data
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify and scale numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    print("Data loaded and scaled.")

    # Handle class imbalance using SMOTE on the training data
    print("\nHandling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("Shape of training data after SMOTE:", X_train_smote.shape)

    # --- Model Training ---
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    results = {}

    print("\nStarting model training and evaluation...")
    for name, model in models.items():
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            "Model": model,
            "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
            "F1-Score": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "Accuracy": accuracy_score(y_test, y_pred)
        }
        
        print(f"\n--- {name} ---")
        print(f"ROC-AUC: {results[name]['ROC-AUC']:.4f}")
        print(f"F1-Score: {results[name]['F1-Score']:.4f}")
        print(f"Precision: {results[name]['Precision']:.4f}")
        print(f"Recall: {results[name]['Recall']:.4f}")

    # --- Save the best model and scaler ---
    best_model_name = max(results, key=lambda name: results[name]['ROC-AUC'])
    best_model = results[best_model_name]['Model']
    
    print(f"\nBest performing model is: {best_model_name} with ROC-AUC of {results[best_model_name]['ROC-AUC']:.4f}")

    joblib.dump(scaler, scaler_path)
    joblib.dump(best_model, model_path)
    
    print(f"\nBest model and scaler have been saved to the '{models_dir}' directory.")

if __name__ == '__main__':
    run_training()
