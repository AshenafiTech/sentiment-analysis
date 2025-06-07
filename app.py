from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("trained_model.sav", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  # Save your vectorizer too

# Preprocess function
stemmer = PorterStemmer()
def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
    return ' '.join(review)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        cleaned = preprocess(user_input)
        vect_input = vectorizer.transform([cleaned])
        result = model.predict(vect_input)[0]
        prediction = "Positive 😊" if result == 1 else "Negative 😞"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
