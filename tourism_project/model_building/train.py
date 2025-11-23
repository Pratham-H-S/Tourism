"""
Model Training Script - Tourism Package Prediction
Trains XGBoost model with hyperparameter tuning
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, roc_auc_score
import joblib
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL TRAINING - TOURISM PACKAGE PREDICTION")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️ HF_TOKEN not set")
    os.environ['HF_TOKEN'] = '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'
    HF_TOKEN = os.getenv("HF_TOKEN")

USERNAME = "DD009"
DATASET_REPO = "DD009/Tourism"
MODEL_REPO = "DD009/tourism-package-model"

api = HfApi(token=HF_TOKEN)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\nStep 1: Loading training data...")

# Try to load from HuggingFace, fallback to local
try:
    print("Attempting to load from HuggingFace...")
    X_train_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="X_train.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
    X_test_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="X_test.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
    y_train_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="y_train.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
    y_test_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="y_test.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
    
    Xtrain = pd.read_csv(X_train_path)
    Xtest = pd.read_csv(X_test_path)
    ytrain = pd.read_csv(y_train_path).squeeze()  # Convert to Series
    ytest = pd.read_csv(y_test_path).squeeze()
    
    print("✓ Data loaded from HuggingFace")
    
except Exception as e:
    print(f"⚠️ HuggingFace load failed: {e}")
    print("Loading from local files...")
    
    # Load from local
    Xtrain = pd.read_csv("X_train.csv")
    Xtest = pd.read_csv("X_test.csv")
    ytrain = pd.read_csv("y_train.csv").squeeze()
    ytest = pd.read_csv("y_test.csv").squeeze()
    
    print("✓ Data loaded from local files")

print(f"Training set: {Xtrain.shape}")
print(f"Test set: {Xtest.shape}")
print(f"Target distribution:\n{ytrain.value_counts()}")

# ============================================================================
# STEP 2: CALCULATE CLASS WEIGHT
# ============================================================================

print("\nStep 2: Handling class imbalance...")

# Calculate class weight for imbalanced data
class_weight = len(ytrain[ytrain == 0]) / len(ytrain[ytrain == 1])
print(f"Class weight (scale_pos_weight): {class_weight:.2f}")

# ============================================================================
# STEP 3: MODEL TRAINING WITH GRID SEARCH
# ============================================================================

print("\nStep 3: Training model with hyperparameter tuning...")
print("This may take several minutes...")

# Define XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'reg_lambda': [0.5, 1.0, 1.5]
}

# Grid Search
print("Starting Grid Search CV...")
grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(Xtrain, ytrain)

print("\n✓ Grid Search completed!")
print(f"\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# ============================================================================
# STEP 4: EVALUATE BEST MODEL
# ============================================================================

print("\nStep 4: Evaluating best model...")

best_model = grid_search.best_estimator_

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

# Metrics
train_metrics = {
    'accuracy': accuracy_score(ytrain, y_pred_train),
    'recall': recall_score(ytrain, y_pred_train),
    'f1_score': f1_score(ytrain, y_pred_train),
    'roc_auc': roc_auc_score(ytrain, y_pred_train_proba)
}

test_metrics = {
    'accuracy': accuracy_score(ytest, y_pred_test),
    'recall': recall_score(ytest, y_pred_test),
    'f1_score': f1_score(ytest, y_pred_test),
    'roc_auc': roc_auc_score(ytest, y_pred_test_proba)
}

print("\nTraining Metrics:")
for metric, value in train_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nTest Metrics:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(ytest, y_pred_test))

# ============================================================================
# STEP 5: SAVE MODEL LOCALLY
# ============================================================================

print("\nStep 5: Saving model...")

os.makedirs('models', exist_ok=True)

# Save model
model_path = "models/best_tourism_model.joblib"
joblib.dump(best_model, model_path)
print(f"✓ Model saved: {model_path}")

# Save model metadata
import json
metadata = {
    'model_type': 'XGBoost',
    'best_params': grid_search.best_params_,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'class_weight': float(class_weight),
    'features': list(Xtrain.columns)
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Metadata saved: models/model_metadata.json")

# Save grid search results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv('models/grid_search_results.csv', index=False)
print("✓ Grid search results saved: models/grid_search_results.csv")

# ============================================================================
# STEP 6: UPLOAD TO HUGGINGFACE
# ============================================================================

print("\nStep 6: Uploading model to HuggingFace...")

try:
    # Check if model repository exists
    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"✓ Repository '{MODEL_REPO}' exists")
    except RepositoryNotFoundError:
        print(f"Creating repository '{MODEL_REPO}'...")
        create_repo(
            repo_id=MODEL_REPO,
            repo_type="model",
            private=False,
            token=HF_TOKEN,
            exist_ok=True
        )
        print(f"✓ Repository created")
    
    # Upload model files
    files_to_upload = [
        ("models/best_tourism_model.joblib", "best_tourism_model.joblib"),
        ("models/model_metadata.json", "model_metadata.json"),
        ("models/label_encoders.pkl", "label_encoders.pkl"),
        ("models/scaler.pkl", "scaler.pkl")
    ]
    
    uploaded = 0
    for local_path, repo_path in files_to_upload:
        try:
            if os.path.exists(local_path):
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=MODEL_REPO,
                    repo_type="model",
                    token=HF_TOKEN
                )
                print(f"  ✓ Uploaded: {repo_path}")
                uploaded += 1
            else:
                print(f"  ⚠️ Not found: {local_path}")
        except Exception as e:
            print(f"  ❌ Failed to upload {repo_path}: {e}")
    
    if uploaded > 0:
        print(f"\n✓ Model uploaded to HuggingFace!")
        print(f"🔗 View model: https://huggingface.co/{MODEL_REPO}")

except Exception as e:
    print(f"\n⚠️ Upload failed: {e}")
    print("Model is saved locally and can be uploaded manually")

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 Best Model Performance:")
print(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

print(f"\n📁 Files saved:")
print(f"  • models/best_tourism_model.joblib")
print(f"  • models/model_metadata.json")
print(f"  • models/grid_search_results.csv")

print(f"\n✅ Ready for deployment!")
print("="*80)
