import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv("comments.csv",encoding='utf-8')
    print(data)
    dt=pd.DataFrame(data)
    clean=dt['CONTENT'].str.replace('[^a-zA-Z\s]+|X{2,}', ' ')
    print(dt)
    dt.head()
    vectorizer = CountVectorizer()
    count = vectorizer.fit_transform(dt['CONTENT'].values)
    classifier = MultinomialNB()
    target = dt['CLASS'].values
    classifier.fit(count,target)
    count=(request.form.values())
    example_count=vectorizer.transform(count)
    predictions=classifier.predict(example_count)
    return render_template('index.html', prediction_text=predictions)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array((data.values()))])

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)