import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CHANGE THIS TO YOUR SMS DATASET NAME
DATA_FILE = os.path.join(BASE_DIR, "csv", "spam.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD DATASET
# =====================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE, encoding="latin-1")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# =====================================
# HANDLE COMMON SPAM DATASET FORMAT
# =====================================

df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# =====================================
# FEATURES & TARGET
# =====================================

X = df["message"]
y = df["label"]

# =====================================
# TF-IDF
# =====================================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)

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
# TRAIN MODEL
# =====================================

print("Training model...")

model = MultinomialNB()

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

with open(
    os.path.join(
        OUTPUT_DIR,
        "sms_spam_model.pkl"
    ),
    "wb"
) as f:
    pickle.dump(model, f)

with open(
    os.path.join(
        OUTPUT_DIR,
        "tfidf_vectorizer.pkl"
    ),
    "wb"
) as f:
    pickle.dump(vectorizer, f)

print("Model Saved Successfully!")

# =====================================
# SAMPLE PREDICTION
# =====================================

sample_message = [
    "Congratulations! You have won a free iPhone. Click here now."
]

sample_vector = vectorizer.transform(sample_message)

prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("\n🚨 SPAM MESSAGE")
else:
    print("\n✅ LEGITIMATE MESSAGE")

print("\n🎉 SMS Spam Detection Project Completed!")