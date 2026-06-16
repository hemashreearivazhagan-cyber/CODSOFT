import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(BASE_DIR, "csv", "Train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD DATASET
# =====================================

print("Loading dataset...")

df = pd.read_csv(TRAIN_FILE)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# =====================================
# DROP UNNECESSARY COLUMNS
# =====================================

drop_cols = [
    "trans_date_trans_time",
    "dob"
]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# =====================================
# HANDLE NULL VALUES
# =====================================

df.fillna(0, inplace=True)

# =====================================
# CONVERT ALL NON-NUMERIC COLUMNS
# =====================================

print("Encoding categorical columns...")

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.factorize(df[col].astype(str))[0]

print("Encoding completed!")

# =====================================
# CHECK DATATYPES
# =====================================

print("\nChecking Data Types...")
print(df.dtypes)

# =====================================
# FEATURES AND TARGET
# =====================================

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Force all columns to numeric

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# MODEL
# =====================================

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# =====================================
# EVALUATION
# =====================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print(f"Accuracy : {accuracy * 100:.2f}%")
print("================================\n")

print(classification_report(y_test, y_pred))

# =====================================
# SAVE MODEL
# =====================================

model_file = os.path.join(
    OUTPUT_DIR,
    "fraud_detection_model.pkl"
)

with open(model_file, "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully!")

# =====================================
# SAVE PREDICTIONS
# =====================================

results = pd.DataFrame({
    "Actual": y_test.iloc[:1000].values,
    "Predicted": y_pred[:1000]
})

prediction_file = os.path.join(
    OUTPUT_DIR,
    "predictions.csv"
)

results.to_csv(prediction_file, index=False)

print("Predictions Saved Successfully!")

# =====================================
# SAMPLE PREDICTION
# =====================================

sample = X_test.iloc[[0]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\n⚠ Fraudulent Transaction Detected")
else:
    print("\n✅ Legitimate Transaction")

print("\n🎉 Credit Card Fraud Detection Project Completed!")