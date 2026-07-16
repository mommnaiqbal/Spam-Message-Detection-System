"""
preprocess.py

This module contains functions for preprocessing SMS text before
training or making predictions.

Techniques Used:
- Lowercasing
- Removing punctuation
- Tokenization
- Stopword removal
- Stemming
"""

import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK resources (runs only if not already downloaded)
nltk.download("punkt")
nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Cleans an SMS message.

    Parameters:
        text (str): Input message

    Returns:
        str: Cleaned message
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords and stem
    cleaned_words = []

    for word in words:
        if word not in stop_words and word.isalpha():
            cleaned_words.append(stemmer.stem(word))

    return " ".join(cleaned_words)


if __name__ == "__main__":

    sample = "Congratulations! You have WON a FREE iPhone!!! Click here now."

    print("Original:")
    print(sample)

    print("\nProcessed:")
    print(clean_text(sample))
