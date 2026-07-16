"""
evaluate.py 

Evaluates the trained Spam Detection model by
displaying:
- Accuracy
- Classification Report
- Confusion Matrix
- Confusion Matrix Visualization
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from preprocess import clean_text

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only required columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# -----------------------------
# Preprocess messages
# -----------------------------
df["clean_message"] = df["message"].apply(clean_text)

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("vectorizer/tfidf_vectorizer.pkl")

# -----------------------------
# Transform text
# -----------------------------
X = vectorizer.transform(df["clean_message"])
y = df["label"]

# -----------------------------
# Predict
# -----------------------------
predictions = model.predict(X)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y, predictions)

print("=" * 50)
print("Model Accuracy")
print("=" * 50)
print(f"{accuracy * 100:.2f}%")

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report\n")

print(classification_report(y, predictions))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y, predictions)

print("Confusion Matrix\n")
print(cm)

# -----------------------------
# Plot Confusion Matrix
# -----------------------------
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues")

plt.title("Spam Detection Confusion Matrix")

plt.show()
