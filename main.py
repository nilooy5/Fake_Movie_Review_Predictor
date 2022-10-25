import glob

import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))


# read all files in the folder data/real_reviews and store them in a list

def read_files(text_files_dir):
    path = text_files_dir
    files = glob.glob(os.path.join(path, '*.txt'))
    real_reviews = []
    for file in files:
        with open(file, 'r') as f:
            real_reviews.append(f.read())

    return real_reviews


reviews_list = read_files('data/real_reviews')
# convert it to a pandas dataframe
reviews_real = pd.DataFrame(reviews_list, columns=['text'])


reviews_generated = pd.read_csv('data/generated_reviews_500.csv')
# drop all columns except the full_text column
reviews_generated = reviews_generated.drop(['prompt_index',
                                            'prompt_text',
                                            'completion_index',
                                            'completion',
                                            'reached_end'],
                                           axis=1)
# rename the full_text column to text
reviews_generated = reviews_generated.rename(columns={'full_text': 'text'})


reviews_real['label'] = 1
print(reviews_real.head())
print(len(reviews_real))

reviews_generated['label'] = 0
print(reviews_generated.head())
print(len(reviews_generated))


# merge the two dataframes
reviews = pd.concat([reviews_real, reviews_generated], ignore_index=True)
print(reviews.head())
# random shuffle the dataframe
reviews = reviews.sample(frac=1).reset_index(drop=True)
print(reviews.head())

# perform cleanup in the text column
# remove all new line characters
reviews['text'] = reviews['text'].str.replace('\n', ' ')
# remove all non-alphabetic characters
reviews['text'] = reviews['text'].str.replace('[^a-zA-Z]', ' ')
# convert all text to lowercase
reviews['text'] = reviews['text'].str.lower()
# remove all stopwords
reviews['text'] = reviews['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# run naive bayes on the data
# split the data into train and test

X_train, X_test, y_train, y_test = train_test_split(reviews['text'], reviews['label'], test_size=0.2, random_state=42)

# vectorize the text
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# train the model

model = MultinomialNB()
model.fit(X_train, y_train)

# evaluate the model

y_pred = model.predict(X_test)
print('========= Naive Bayes =========')
print('Accuracy: ')
print(accuracy_score(y_test, y_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
print('Classification Report: ')
print(classification_report(y_test, y_pred))
print('======================================================')
print()


# run svm on the data
# train the model

model = SVC()
model.fit(X_train, y_train)

# evaluate the model
y_pred = model.predict(X_test)
print('========= SVM =========')
print('Accuracy: ')
print(accuracy_score(y_test, y_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
print('Classification Report: ')
print(classification_report(y_test, y_pred))
print('======================================================')
print()

# run logistic regression on the data
# train the model

model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate the model
y_pred = model.predict(X_test)
print('========= Logistic Regression =========')
print('Accuracy: ')
print(accuracy_score(y_test, y_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
print('Classification Report: ')
print(classification_report(y_test, y_pred))
print('======================================================')
print()

# run random forest on the data
# train the model

model = RandomForestClassifier()
model.fit(X_train, y_train)

# evaluate the model
y_pred = model.predict(X_test)
print('========= Random Forest =========')
print('Accuracy: ')
print(accuracy_score(y_test, y_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
print('Classification Report: ')
print(classification_report(y_test, y_pred))
print('======================================================')
print()
