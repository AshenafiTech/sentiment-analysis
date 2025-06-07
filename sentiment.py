# %%
import numpy as np
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datasets import load_dataset

# %%
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()

# %%
# Load the sentiment split from TweetEval
dataset = load_dataset("tweet_eval", "sentiment")
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()
val_df = dataset['validation'].to_pandas()

# Combine all splits
twitter_data = pd.concat([train_df, test_df, val_df], ignore_index=True)

# Map numeric labels to text (optional)
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
twitter_data['label_text'] = twitter_data['label'].map(label_map)
twitter_data = twitter_data.rename(columns={"text": "text", "label": "target"})

# Shuffle
twitter_data = twitter_data.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
# Stemming + preprocessing
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

# %%
# Prepare features and labels
X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

# %%
# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# %%
# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# %%
# Logistic Regression Model
model = LogisticRegression(max_iter=100)
model.fit(X_train, Y_train)

# %%
# Accuracy scores
X_train_pred = model.predict(X_train)
X_test_pred = model.predict(X_test)

train_acc = accuracy_score(Y_train, X_train_pred)
test_acc = accuracy_score(Y_test, X_test_pred)

print(f"✅ Training Accuracy: {train_acc}")
print(f"✅ Testing Accuracy: {test_acc}")

# %%
# Save model + vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

# %%
# Load model + vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    data = pickle.load(f)
    loaded_model = data['model']
    vectorizer = data['vectorizer']

# %%
# Inference function
def predict_sentiment(input_text):
    stemmed = re.sub('[^a-zA-Z]', ' ', input_text).lower().split()
    stemmed = [port_stem.stem(word) for word in stemmed if word not in stop_words]
    final_text = ' '.join(stemmed)

    vectorized_input = vectorizer.transform([final_text])
    prediction = loaded_model.predict(vectorized_input)[0]

    label_map = {0: "❌ Negative", 1: "😐 Neutral", 2: "✅ Positive"}
    sentiment = label_map.get(prediction, "Unknown")

    print(f"Input: {input_text}")
    print(f"Predicted Sentiment: {sentiment}")

# %%
# 🧪 Example Usage
predict_sentiment("I like Ethiopia")


# %%
