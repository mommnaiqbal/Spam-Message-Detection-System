"""
train_model.py

This script:
1. Loads the dataset
2. Cleans the text
3. Converts text into TF-IDF features
4. Splits the dataset
5. Trains a Multinomial Naive Bayes model
6. Evaluates the model
7. Saves the trained model and vectorizer
"""

import os
import joblib
import pandas as pd

from preprocess import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -------------------------
# Create folders if needed
# -------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("vectorizer", exist_ok=True)


# -------------------------
# Load Dataset
# -------------------------
# Expected dataset:
# data/spam.csv
#
# Columns:
# label,message

df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only required columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# -------------------------
# Clean Messages
# -------------------------
df["clean_message"] = df["message"].apply(clean_text)

# -------------------------
# TF-IDF
# -------------------------
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["clean_message"])

y = df["label"]

# -------------------------
# Split Dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------
model = MultinomialNB()

model.fit(X_train, y_train)

# -------------------------
# Prediction
# -------------------------
predictions = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print("Accuracy")
print("=" * 50)

print(f"{accuracy * 100:.2f}%")

print("\n")

print("=" * 50)
print("Classification Report")
print("=" * 50)

print(classification_report(y_test, predictions))

print("=" * 50)
print("Confusion Matrix")
print("=" * 50)

print(confusion_matrix(y_test, predictions))

# -------------------------
# Save Model
# -------------------------
joblib.dump(
    model,
    "models/spam_classifier.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully!")
print("Vectorizer saved successfully!")
