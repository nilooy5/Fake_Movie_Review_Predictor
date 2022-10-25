
# run bert on the data
# train the model

from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# split reviews['text'] and reviews['label'] into train and test
X_train, X_test, y_train, y_test = train_test_split(reviews['text'], reviews['label'], test_size=0.2, random_state=42)

# make train and test data from reviews['text'] and reviews['label']
# convert the text to ids
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

# convert the labels to tensors
train_labels = tf.convert_to_tensor(list(y_train))
test_labels = tf.convert_to_tensor(list(y_test))

# create a tf dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
))

# train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                loss=model.compute_loss,  # can also use any keras loss fn
                metrics=['accuracy'])

model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16,
            validation_data=test_dataset.shuffle(1000).batch(16))

# evaluate the model
model.evaluate(test_dataset.batch(16))

#
#
# # tokenize the data
# X_train = tokenizer(X_train, padding=True, truncation=True, return_tensors='tf')
# X_test = tokenizer(X_test, padding=True, truncation=True, return_tensors='tf')
#
# # train the model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
#                 loss=model.compute_loss,
#                 metrics=['accuracy'])
#
# model.fit(X_train, y_train, epochs=3, batch_size=16)
#
# # evaluate the model
# model.evaluate(X_test, y_test)
#
# # run roberta on the data
# # train the model
