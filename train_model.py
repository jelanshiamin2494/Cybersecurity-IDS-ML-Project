import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # Added this
from sklearn.metrics import accuracy_score

def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    X_path = os.path.join(project_root, 'dataset', 'X.csv')
    y_path = os.path.join(project_root, 'dataset', 'y.csv')
    
    
    rf_save_path = os.path.join(project_root, 'models', 'random_forest_model.joblib')
    lr_save_path = os.path.join(project_root, 'models', 'logistic_regression_model.joblib')

    print("--- ML Training Pipeline Started ---")

    try:
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).squeeze()
        
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)
        y = y[X.index] 

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, rf_save_path)
        print(f"Random Forest saved to: {rf_save_path}")

        
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, solver='liblinear') # Standard for this dataset
        lr_model.fit(X_train, y_train)
        joblib.dump(lr_model, lr_save_path)
        print(f"Logistic Regression saved to: {lr_save_path}")

        print("\n SUCCESS: Both models trained and saved!")

    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    train()