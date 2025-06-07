# Table of Contents

## 1. Sentiment Analysis AI Model

1.1 [Project Overview](#project-overview)  
1.2 [Dataset Source](#dataset-source)  
1.3 [Step 1 – Dataset Extraction](#step-1--dataset-extraction)  
1.4 [Step 2 – Import Dependencies](#step-2--import-dependencies)  
1.5 [Step 3 – Load and Inspect the Dataset](#step-3--load-and-inspect-the-dataset)  
1.6 [Step 4 – Simplify Sentiment Labels](#step-4--simplify-sentiment-labels)  
1.7 [Step 5 – Preprocess the Text Data](#step-5--preprocess-the-text-data)  
1.8 [Step 6 – Define Features and Labels](#step-6--define-features-and-labels)  
1.9 [Step 7 – Train-Test Split](#step-7--train-test-split)  
1.10 [Step 8 – TF-IDF Vectorization](#step-8--tf-idf-vectorization)  
1.11 [Step 9 – Train Logistic Regression Model](#step-9--train-logistic-regression-model)  
1.12 [Step 10 – Evaluate Model Accuracy](#step-10--evaluate-model-accuracy)  
1.13 [Step 11 – Save and Reload Model](#step-11--save-and-reload-model)  
1.14 [Step 12 – Predict Sentiment for Custom Input](#step-12--predict-sentiment-for-custom-input)  
1.15 [Output Example](#output-example)  

---

## 2. Flask Application

2.1 [Overview](#overview)  
2.2 [Project Structure](#project-structure)  
2.3 [Requirements](#requirements)  
2.4 [How to Run the Application](#how-to-run-the-application)  
2.5 [Application Features](#application-features)  
2.6 [Sentiment Analysis Pipeline](#sentiment-analysis-pipeline)  
2.7 [Files Explained](#files-explained)  
2.8 [Notes](#notes)  
2.9 [Security Tips for Production](#security-tips-for-production)  
2.10 [Example Use Case](#example-use-case)  
2.11 [Author & License](#author--license)  

# Sentiment Analysis with Logistic Regression – Detailed Project Documentation


This project performs binary sentiment classification (positive or negative) on tweets using the Sentiment140 dataset. It covers the entire pipeline from data extraction to model deployment and prediction.

---

we used a data from a https://www.kaggle.com/datasets/kazanova/sentiment140 and the dataset is available in a zip file format. you can download it from the link above and place it in the same directory as this notebook. it's named `sentiment140.zip`. The dataset contains 1.6 million tweets labeled with sentiment (0 for negative, 4 for positive).

## Step 1 – Dataset Extraction

**Description:**
The dataset is provided in a ZIP format. This step extracts the compressed dataset so that it can be accessed as a CSV file.


```python
from zipfile import ZipFile
dataset = './sentiment140.zip'
with ZipFile(dataset, 'r') as zip:
    zip.extractall()
```

* `ZipFile` is used to open the ZIP archive.
* `.extractall()` unpacks all contents of the archive into the current directory.
* the output of this step is a CSV file named `training.1600000.processed.noemoticon.csv` which contains the tweets and their sentiment labels.

---

## Step 2 – Import Dependencies

**Description:**
This step imports all required Python libraries for data preprocessing, NLP, modeling, and evaluation.

```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
```

* `pandas` and `numpy`: for data loading and numerical operations.
* `re`: for regular expressions used in text cleaning.
* `nltk`: for natural language processing like stopword removal and stemming.
* `sklearn`: for machine learning operations such as feature extraction, model training, and evaluation.

---

## Step 3 – Load and Inspect the Dataset

**Description:**
This step loads the dataset into a DataFrame and assigns column names for easier reference.

```python
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('./training.1600000.processed.noemoticon.csv', names=column_names, encoding='ISO-8859-1')
```

* `names=column_names` assigns custom column headers to the CSV.
* `encoding='ISO-8859-1'` ensures special characters in the tweets are handled correctly.

---

## Step 4 – Simplify Sentiment Labels

**Description:**
Sentiment140 uses 0 for negative and 4 for positive tweets. This step converts 4 to 1 for binary classification (0 = Negative, 1 = Positive).

```python
twitter_data['target'] = twitter_data['target'].replace(4, 1)
```

* This makes the sentiment binary, simplifying the classification task.

---

## Step 5 – Preprocess the Text Data

**Description:**
Tweets are cleaned and standardized to remove noise and prepare for feature extraction. This includes removing punctuation, converting to lowercase, removing stopwords, and applying stemming.

```python
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)
```

* `re.sub('[^a-zA-Z]', ' ', content)`: Removes non-alphabetic characters.
* `lower().split()`: Converts to lowercase and splits into words.
* `stopwords.words('english')`: Removes common words with low semantic value.
* `PorterStemmer`: Reduces words to their root form (e.g., “running” becomes “run”).

---

## Step 6 – Define Features and Labels

**Description:**
The cleaned text and corresponding sentiment labels are extracted for training the model.

```python
X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values
```

* `X`: Input features (processed tweets).
* `Y`: Output labels (0 or 1 indicating sentiment).

---

## Step 7 – Train-Test Split

**Description:**
Splits the data into training and test sets for evaluating model performance.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
```

* `test_size=0.2`: 20% of the data is reserved for testing.
* `stratify=Y`: Maintains equal proportion of positive and negative samples in both sets.
* `random_state=2`: Ensures reproducible results.

---

## Step 8 – TF-IDF Vectorization

**Description:**
Text data is transformed into numerical feature vectors using the TF-IDF technique, which reflects how important a word is in a document relative to the entire corpus.

```python
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
```

* `fit()`: Learns the vocabulary from the training data.
* `transform()`: Converts text to sparse numerical vectors based on learned vocabulary.

---

## Step 9 – Train Logistic Regression Model

**Description:**
A logistic regression model is trained using the vectorized text data.

```python
model = LogisticRegression(max_iter=100)
model.fit(X_train, Y_train)
```

* `LogisticRegression`: A linear classifier suitable for binary classification.
* `max_iter=100`: Sets the maximum number of optimization iterations.

---

## Step 10 – Evaluate Model Accuracy

**Description:**
Model performance is evaluated on both the training and test datasets using accuracy.

```python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Training Accuracy:', training_data_accuracy)
print('Test Accuracy:', test_data_accuracy)
```

* `accuracy_score`: Compares predicted values with true labels.
* Helps detect overfitting if training accuracy is much higher than test accuracy.

---

## Step 11 – Save and Reload Model

**Description:**
The trained model is saved to disk using `pickle` and can be loaded back later for prediction without retraining.

```python
import pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
```

* `pickle.dump`: Serializes and saves the model.
* `pickle.load`: Loads the saved model back into memory.

---

## Step 12 – Predict Sentiment for Custom Input

**Description:**
This function allows real-time predictions for new input texts using the same pipeline (preprocessing + vectorization + prediction).

```python
def predict_sentiment(text):
    processed = stemming(text)
    vectorized = vectorizer.transform([processed])
    prediction = loaded_model.predict(vectorized)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Example
print(predict_sentiment("I love this app. It's amazing!"))
print(predict_sentiment("Worst experience ever."))
```

* `stemming(text)`: Applies same preprocessing as training data.
* `vectorizer.transform`: Converts input to the same TF-IDF space.
* `loaded_model.predict`: Returns prediction label.

---

## Output Example

```
Training Accuracy: 0.89
Test Accuracy: 0.86
Positive
Negative
```

---
# Flask Application Documentation

## Overview

The **AI Sentiment Analyzer** is a web application built with Flask that uses a pre-trained machine learning model to analyze the sentiment of user-input text. It classifies the sentiment as either **Positive** or **Negative**.

---

## Project Structure

```
sentiment_app/
├── static/
│   └── style.css              # Custom CSS styling for the AI interface
├── templates/
│   └── index.html             # HTML template rendered by Flask
├── app.py                     # Main Flask application
├── trained_model.sav          # Pickled trained machine learning model
└── vectorizer.pkl             # Pickled text vectorizer (e.g., CountVectorizer or TfidfVectorizer)
```

---

## Requirements

Ensure the following Python packages are installed:

```bash
pip install flask nltk scikit-learn
```

Additionally, download the required NLTK resources:

```python
import nltk
nltk.download('stopwords')
```

---

## How to Run the Application

1. Navigate to the project folder:

   ```bash
   cd sentiment_app
   ```

2. Start the Flask application:

   ```bash
   python app.py
   ```

3. Open the application in your web browser:

   ```
   http://127.0.0.1:5000
   ```

---

## Application Features

* **User Input**: Users can type a sentence or paragraph into a text area.
* **Sentiment Prediction**: Upon submission, the application returns a sentiment classification.
* **Modern UI**: The interface is styled with Bootstrap and custom CSS to resemble a modern AI interface.

---

## Sentiment Analysis Pipeline

### 1. Preprocessing

Text input is cleaned and normalized through the following steps:

* Remove non-alphabetic characters.
* Convert to lowercase.
* Tokenize and remove English stopwords.
* Apply stemming using NLTK's `PorterStemmer`.

### 2. Vectorization

The processed text is transformed into a numeric format using the saved `vectorizer.pkl`.

### 3. Prediction

The model (`trained_model.sav`) outputs:

* `1` → Positive
* `0` → Negative

---

## Files Explained

### `app.py`

Contains the main logic for serving the web app, handling requests, and performing sentiment prediction.

### `index.html`

Defines the layout and form for the frontend using Bootstrap and Jinja2 templating. Displays prediction results.

### `style.css`

Custom styles for a dark-themed, AI-inspired user interface.

---

## Notes

* Ensure the saved model and vectorizer are consistent with the preprocessing pipeline used in the app.
* For production, disable `debug=True` and deploy using a WSGI server such as Gunicorn or uWSGI.

---

## Security Tips for Production

* Sanitize and validate all user inputs.
* Use CSRF protection for forms.
* Limit request size to mitigate potential overload or abuse.

---

## Example Use Case

**Input:**

```
I absolutely loved the new movie. It was fantastic!
```

**Output:**

```
Sentiment: Positive
```

---

## Author & License

Developed by \[Jiru Gutema, Ashenafi Godana, Abigya Daniel, Kalafat]
License: MIT (or another appropriate license)
