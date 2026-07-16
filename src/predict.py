"""
predict.py

Loads the trained model and TF-IDF vectorizer,
preprocesses a user's message, and predicts
whether it is Spam or Ham.
"""

import joblib

from preprocess import clean_text

# Load saved model
model = joblib.load("models/spam_classifier.pkl")

# Load saved TF-IDF vectorizer
vectorizer = joblib.load("vectorizer/tfidf_vectorizer.pkl")


def predict_message(message):
    """
    Predict whether a message is Spam or Ham.

    Parameters:
        message (str): SMS message

    Returns:
        str: Prediction ("Spam" or "Ham")
    """

    # Clean the input text
    cleaned = clean_text(message)

    # Convert to TF-IDF features
    vector = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "Spam"
    else:
        return "Ham"


def main():
    print("=" * 50)
    print("Spam Message Detection System")
    print("=" * 50)

    while True:
        message = input("\nEnter an SMS message:\n")

        result = predict_message(message)

        print("\nPrediction:", result)

        again = input("\nDo you want to test another message? (y/n): ")

        if again.lower() != "y":
            print("\nThank you for using the Spam Message Detection System!")
            break


if __name__ == "__main__":
    main()
