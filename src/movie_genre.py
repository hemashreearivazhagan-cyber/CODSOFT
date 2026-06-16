import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = os.path.join(BASE_DIR, "csv", "train_data.csv")
TEST_FILE = os.path.join(BASE_DIR, "csv", "test_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD TRAINING DATA
# =====================================

print("Loading training data...")

train_df = pd.read_csv(
    TRAIN_FILE,
    sep=" ::: ",
    engine="python",
    names=["id", "title", "genre", "description"]
)

print("Training data loaded successfully!")
print("Total Records:", len(train_df))

# =====================================
# PREPARE DATA
# =====================================

X = train_df["description"].fillna("")
y = train_df["genre"]

# =====================================
# TF-IDF VECTORIZATION
# =====================================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
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

print("Model trained successfully!")

# =====================================
# EVALUATE MODEL
# =====================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print(f"Accuracy : {accuracy * 100:.2f}%")
print("====================================\n")

print(classification_report(y_test, y_pred))

# =====================================
# SAVE MODEL
# =====================================

model_path = os.path.join(
    OUTPUT_DIR,
    "movie_genre_model.pkl"
)

vectorizer_path = os.path.join(
    OUTPUT_DIR,
    "tfidf_vectorizer.pkl"
)

pickle.dump(model, open(model_path, "wb"))
pickle.dump(vectorizer, open(vectorizer_path, "wb"))

print("Model saved successfully!")

# =====================================
# LOAD TEST DATA
# =====================================

print("Loading test data...")

test_df = pd.read_csv(
    TEST_FILE,
    sep=" ::: ",
    engine="python",
    names=["id", "title", "description"]
)

# =====================================
# PREDICT GENRES
# =====================================

X_new = vectorizer.transform(
    test_df["description"].fillna("")
)

predictions = model.predict(X_new)

# =====================================
# SAVE PREDICTIONS
# =====================================

prediction_file = os.path.join(
    OUTPUT_DIR,
    "predictions.csv"
)

result = pd.DataFrame({
    "id": test_df["id"],
    "Predicted_Genre": predictions
})

result.to_csv(
    prediction_file,
    index=False
)

print("\nPredictions saved successfully!")
print("Output File:", prediction_file)

print("\nProject Completed Successfully!")