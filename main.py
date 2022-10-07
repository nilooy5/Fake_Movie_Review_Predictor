import glob

import pandas as pd
import os
import nltk
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

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
