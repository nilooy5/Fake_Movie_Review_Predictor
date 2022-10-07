import glob

import pandas as pd
import os


# read all files in the folder data/real_reviews and store them in a list

def read_files(text_files_dir):
    path = text_files_dir
    files = glob.glob(os.path.join(path, '*.txt'))
    real_reviews = []
    for file in files:
        with open(file, 'r') as f:
            real_reviews.append(f.read())

    return real_reviews


reviews_real = read_files('data/real_reviews')
# convert it to a pandas dataframe
df_real = pd.DataFrame(reviews_real, columns=['text'])
df_real['label'] = 1
print(df_real.head())
print(len(df_real))


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

reviews_generated['label'] = 0
print(reviews_generated.head())
# length of the generated reviews
print(len(reviews_generated))
