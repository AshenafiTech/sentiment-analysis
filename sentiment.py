from zipfile import ZipFile
dataset = './sentiment140.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')


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



twitter_data = pd.read_csv('./training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')

twitter_data.shape

twitter_data.head()

column_names = ['target', 'id', 'data', 'flag', 'user', 'text']
twitter_data = pd.read_csv('./training.1600000.processed.noemoticon.csv', names=column_names, encoding='ISO-8859-1')

twitter_data.shape


twitter_data['target'] = twitter_data['target'].replace(4,1)

twitter_data['target'].value_counts()





port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content





nltk.download('stopwords')


twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

twitter_data.head()

print(twitter_data['stemmed_content'])

print(twitter_data['target'])

X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

print(X)

print(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

print(X_test)


vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train)

print(X_test)



model = LogisticRegression(max_iter=100)
model.fit(X_train, Y_train)



X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on the training data: ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
training_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on the test data: ', training_data_accuracy)


filename = 'trained_model.sav'

import pickle


pickle.dump(model, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))

X_new = X_test[3]
print(Y_test[3])

prediction = loaded_model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('Negative Tweet')

else:
  print('Positive Tweet')


import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

port_stem = PorterStemmer()

def predict_sentiment(input_text):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', input_text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    final_text = ' '.join(stemmed_content)

    vectorized_input = vectorizer.transform([final_text])
    prediction = loaded_model.predict(vectorized_input)

    sentiment = "✅ Positive" if prediction[0] == 1 else "❌ Negative"
    print(f"Input: {input_text}")
    print(f"Predicted Sentiment: {sentiment}")

predict_sentiment("I liked this product")
predict_sentiment("This is the worst experience ever")




