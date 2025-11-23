# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder, StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, hf_hub_download
import joblib
import pickle
# os.environ['HF_TOKEN'] = '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'
# HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN = HfApi(token=os.getenv("HF_TOKEN"))
# api = HfApi(token=HF_TOKEN)

# FIX: Try multiple ways to load the data
try:
    # Method 1: Try direct download (most reliable)
    print("Attempting to download from HuggingFace...")
    file_path = hf_hub_download(
        repo_id="DD009/Tourism",
        filename="tourism.csv",  # Try lowercase first
        repo_type="dataset",
        token=HF_TOKEN
    )
    df = pd.read_csv(file_path)
    print("✓ Dataset loaded from HuggingFace (tourism.csv)")
except:
    try:
        # Method 2: Try uppercase
        file_path = hf_hub_download(
            repo_id="DD009/Tourism",
            filename="Tourism.csv",
            repo_type="dataset",
            token=HF_TOKEN
        )
        df = pd.read_csv(file_path)
        print("✓ Dataset loaded from HuggingFace (Tourism.csv)")
    except:
        # Method 3: Load from local
        print("⚠️ HuggingFace load failed, trying local file...")
        df = pd.read_csv("tourism.csv")
        print("✓ Dataset loaded from local file")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Encode ALL categorical columns
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Save encoders
os.makedirs('models', exist_ok=True)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# CRITICAL FIX: Use ProdTaken as target, not MonthlyIncome!
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Add feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
joblib.dump(scaler, 'models/scaler.pkl')

# Perform train-test split with stratification
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save with better naming
Xtrain.to_csv("X_train.csv", index=False)
Xtest.to_csv("X_test.csv", index=False)
ytrain.to_csv("y_train.csv", index=False)
ytest.to_csv("y_test.csv", index=False)

files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

# Upload with error handling
for file_path in files:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id="DD009/Tourism",
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"✓ Uploaded: {file_path}")
    except Exception as e:
        print(f"⚠️ Upload failed for {file_path}: {e}")

print("\n✓ Data preparation complete!")
