import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os
import glob

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# Load Datasets from clean_data directory
files = glob.glob("clean_data/*.csv")
dfs = []
for file in files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"Loaded {file}: {df.shape}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

if not dfs:
    print("No data files loaded from clean_data.")
    exit()

data = pd.concat(dfs, ignore_index=True)
print(f"Total Combined Data: {data.shape}")

# Features and Target
# Using full feature set as per assignment requirements (12 features)
features = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
target = 'FTR' # Full Time Result: H, D, A

# Drop Rows with Missing Values in Features
data = data.dropna(subset=features + [target])
print(f"Data after dropping NA: {data.shape}")

X = data[features]
y = data[target]

# Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'models/label_encoder.pkl')
print(f"Classes: {le.classes_}") # ['A', 'D', 'H'] usually 0, 1, 2

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

results = []

print("\nTraining and Evaluating Models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    # AUC for multi-class needs One-vs-Rest (ovr) or One-vs-One (ovo)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })
    
    # Save Model
    filename = f"models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"Saved {name} to {filename}")

# Create DataFrame for Comparison Table
results_df = pd.DataFrame(results)
print("\nComparison Table:")
print(results_df.to_markdown(index=False))

# Make sure to save the results so we can put them in README
results_df.to_csv("model_comparison.csv", index=False)
