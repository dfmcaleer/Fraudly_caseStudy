import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import cPickle as pickle

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

from sklearn.feature_extraction import text
from numpy.linalg import lstsq
from sklearn.cross_validation import train_test_split

import requests
import json
import pandas as pd

# from topic_dummies import get_data

def predict(one_line):
    #Expects one line df of new data

    def get_text(cell):
        return BeautifulSoup(cell, 'html.parser').get_text()

    clean_text = one_line['description'].apply(get_text)

    with open('vectorizer3.pkl') as f:
        vectorizer = pickle.load(f)
    with open('model3.pkl') as f:
        model = pickle.load(f)

    def make_prediction(text):
        if not text.empty:
            X = vectorizer.transform(text)
            prediction = model.predict(X)[0]
            return prediction
    prediction = make_prediction(clean_text)

    one_line['predicted_topic'] = prediction

    return one_line


r = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
data=json.loads(r.text)
one_line = pd.Series(data,index=data.keys()).to_frame().T


print predict(one_line)
