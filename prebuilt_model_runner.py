import pandas as pd
import glob
import os
from nltk.corpus import stopwords
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
stop_words = set(stopwords.words('english'))

def read_files(text_files_dir):
    path = text_files_dir
    files = glob.glob(os.path.join(path, '*.txt'))
    real_reviews = []
    for file in files:
        with open(file, 'r') as f:
            real_reviews.append(f.read())

    return real_reviews


reviews_list = read_files('unseen_data')
# convert it to a pandas dataframe
reviews = pd.DataFrame(reviews_list, columns=['text'])

reviews['label'] = 1



# perform cleanup in the text column
# remove all new line characters
reviews['text'] = reviews['text'].str.replace('\n', ' ')
# remove all non-alphabetic characters
reviews['text'] = reviews['text'].str.replace('[^a-zA-Z]', ' ')
# convert all text to lowercase
reviews['text'] = reviews['text'].str.lower()
# remove all stopwords
reviews['text'] = reviews['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

X_train, X_test, y_train, y_test = train_test_split(reviews['text'], reviews['label'], test_size=1, random_state=10)

# vectorize the text
vectorizer = TfidfVectorizer()
# X_train = vectorizer.transform(X_train).toarray()
X_test = vectorizer.fit_transform(X_test).toarray()
print(X_test)

with open('modelRF.pkl', 'rb') as f:
    modelRFLoaded = pickle.load(f)

# predict the labels
y_pred = modelRFLoaded.predict(X_test)
print(y_pred)


