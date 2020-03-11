import csv
import numpy as np
import tensorflow as tf
import preprocessor as p
import re
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Method to plot graphs
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

def clean_tweets(csv_path):
    with open(csv_path) as csvFile:
        p.set_options(p.OPT.URL, p.OPT.MENTION)
        csv_reader = csv.reader(csvFile, delimiter=",")
        tweets = list(csv_reader)

        labels = []
        examples = []

        for tweet in tweets:
            labels.append(int(tweet[0]))
            clean_tweet = p.clean(tweet[1].lower())
            punctuation_less_tweet = re.sub('[^A-Za-z0-9 ]+', '', clean_tweet)
            examples.append(punctuation_less_tweet)

        return examples, labels

examples, labels = clean_tweets("./data/twitter_training_data.csv")

# Tokenise, encode and pad data
tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()
for text in examples:
    tokens = tokenizer.tokenize(text)
    vocabulary_set.update(tokens)

vocab_size = len(vocabulary_set)
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
encoder.save_to_file('./models/twitter_data_encoder')

encoded_examples = []
for comment in examples:
    encoded_examples.append(encoder.encode(comment))

encoded_examples = tf.keras.preprocessing.sequence.pad_sequences(encoded_examples, maxlen=50, padding='post')

# Split data into training and testing
x_train = encoded_examples[:int(len(encoded_examples) * 0.7)]
y_train = labels[:int(len(encoded_examples) * 0.7)]
x_test = encoded_examples[int(len(encoded_examples) * 0.7):]
y_test = labels[int(len(encoded_examples) * 0.7):]

# Convert data to a tensorflow dataset and shuffle the batches
training_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 10
SHUFFLE_BUFFER_SIZE = 100

training_data = training_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

# Model structure
sentiment_analysis_model = tf.keras.Sequential()
sentiment_analysis_model.add(layers.Embedding(encoder.vocab_size, 64))
sentiment_analysis_model.add(layers.Bidirectional(layers.LSTM(64)))
sentiment_analysis_model.add(layers.Dense(64, activation='relu'))
sentiment_analysis_model.add(layers.Dropout(0.5))
sentiment_analysis_model.add(layers.Dense(4, activation='softmax'))
sentiment_analysis_model.summary()

# Train and test model
sentiment_analysis_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4),
                                 metrics=['accuracy'])
history = sentiment_analysis_model.fit(training_data, epochs=3, validation_data=test_data, validation_steps=30)
sentiment_analysis_model.evaluate(test_data)

# Save model to a file so it can be loaded later on.
tf.keras.models.save_model(sentiment_analysis_model, './models/twitter_dataset_model.h5')