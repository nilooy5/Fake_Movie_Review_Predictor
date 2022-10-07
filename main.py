import glob

import pandas as pd
import os


# read all files in the folder data/real_reviews and store them in a list

def read_files(text_files_dir):
    path = text_files_dir
    files = glob.glob(os.path.join(path, '*.txt'))
    print("list of files: ", len(files))
    real_reviews = []
    for file in files:
        with open(file, 'r') as f:
            real_reviews.append(f.read())

    return real_reviews


reviews_real = read_files('data/real_reviews')
print(len(reviews_real))
print(reviews_real[1])

reviews_generated = pd.read_csv('data/generated_reviews_500.csv',
                                names=['prompt_index',
                                       'prompt_texts',
                                       'completion_text',
                                       'completion',
                                       'full_text',
                                       'reached_end'],
                                skiprows=1)

# length of the generated reviews
print(len(reviews_generated))
# drop all columns except the full_text column
reviews_generated = reviews_generated.drop(['prompt_index', 'prompt_texts', 'completion_text', 'completion', 'reached_end'], axis=1)
# print the first 5 rows
print(reviews_generated.head())
# print column names of the dataframe
print(reviews_generated.columns)


