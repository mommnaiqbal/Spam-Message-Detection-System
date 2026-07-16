# Spam Message Detection System

A Machine Learning project that detects whether an SMS message is **Spam** or **Ham (Not Spam)** using **Python** and **Scikit-learn**. The model uses Natural Language Processing (NLP) techniques including text preprocessing and TF-IDF vectorization, followed by a Multinomial Naive Bayes classifier.

---

## 📌 Features

- Text preprocessing using NLTK
- Stopword removal
- Stemming
- TF-IDF Vectorization
- Multinomial Naive Bayes classifier
- Model training and testing
- Accuracy evaluation
- Confusion Matrix
- Classification Report
- Save and load trained model
- Predict custom SMS messages
- Clean and organized GitHub-ready project structure

---

## 🛠 Technologies Used

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Matplotlib
- Joblib

---

## 📁 Project Structure

```
Spam-Message-Detection-System/
│
├── data/
│   └── spam.csv
│
├── models/
│   └── spam_classifier.pkl
│
├── vectorizer/
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
│
├── notebooks/
│   └── Spam_Detection.ipynb
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

---

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**, a public dataset containing thousands of labeled SMS messages classified as **spam** or **ham**.

Typical dataset columns:

| Label | Message |
|-------|---------|
| ham | Hello, how are you? |
| spam | Congratulations! You've won a free prize. |

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Spam-Message-Detection-System.git
```

Move into the project directory:

```bash
cd Spam-Message-Detection-System
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Project

Train the model:

```bash
python src/train_model.py
```

Run the prediction script:

```bash
python src/predict.py
```

Or run the complete application:

```bash
python main.py
```

---

## 🧠 Machine Learning Pipeline

1. Load Dataset
2. Clean Text
3. Convert to Lowercase
4. Remove Punctuation
5. Remove Stopwords
6. Apply Stemming
7. TF-IDF Vectorization
8. Train Multinomial Naive Bayes Model
9. Evaluate Model
10. Save Model
11. Predict New Messages

---

## 📈 Model Evaluation

The project evaluates the trained model using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1 Score
- Classification Report

Expected model accuracy:

```
97% - 99%
```

(depending on dataset version and train-test split)

---

## 💬 Example Prediction

Input:

```
Congratulations! You have won a FREE iPhone. Click here to claim now!
```

Output:

```
Spam
```

Input:

```
Hey, are we still meeting at 6 PM today?
```

Output:

```
Ham
```

---

## 📦 Dependencies

- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- joblib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

- Flask web application
- Streamlit interface
- Deep Learning (LSTM)
- BERT-based spam detection
- Email spam classification
- REST API deployment
- Docker support

---

## 👨‍💻 Author

Developed as a Machine Learning and Natural Language Processing project using Python and Scikit-learn.

---

## 📄 License

This project is licensed under the MIT License.
