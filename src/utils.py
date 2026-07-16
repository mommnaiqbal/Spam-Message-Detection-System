"""
utils.py

Utility functions used throughout the Spam Message
Detection System.
"""

import joblib
from preprocess import clean_text


def load_model():
    """
    Load the trained Naive Bayes model.

    Returns:
        sklearn model
    """
    return joblib.load("models/spam_classifier.pkl")


def load_vectorizer():
    """
    Load the saved TF-IDF vectorizer.

    Returns:
        TfidfVectorizer
    """
    return joblib.load("vectorizer/tfidf_vectorizer.pkl")


def predict(message):
    """
    Predict whether a message is Spam or Ham.

    Parameters:
        message (str)

    Returns:
        tuple:
            prediction (str)
            probability (float)
    """

    model = load_model()
    vectorizer = load_vectorizer()

    cleaned_message = clean_text(message)

    vector = vectorizer.transform([cleaned_message])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities)

    if prediction == 1:
        label = "Spam"
    else:
        label = "Ham"

    return label, confidence


def print_prediction(message):
    """
    Print a formatted prediction result.
    """

    label, confidence = predict(message)

    print("\n==============================")
    print("Prediction Result")
    print("==============================")
    print(f"Message    : {message}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2%}")
