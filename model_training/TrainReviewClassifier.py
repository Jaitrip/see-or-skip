import csv
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def clean_reviews(csv_path):
    with open(csv_path, encoding='cp850') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=",")
        reviews = list(csv_reader)
        reviews.pop(0)

        labels = []
        examples = []

        for review in reviews:
            labels.append(int(review[0]))
            punctuation_less_tweet = re.sub('[^A-Za-z0-9 ]+', '', review[1].lower())
            examples.append(punctuation_less_tweet)

        return examples, labels

def shuffle_data(x_train, y_train):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    random_shuffle = np.arange(len(x_train))
    np.random.shuffle(random_shuffle)
    shuffled_x_train = x_train[random_shuffle]
    shuffled_y_train = y_train[random_shuffle]

    return shuffled_x_train, shuffled_y_train

# Data import and preprocessing
dataset = "../data/rotten_tomatoes_reviews.csv"
CLASSES = ["Negative", "Positive"]

examples, labels = clean_reviews(dataset)

# Tokenise, encode and pad data
tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()
for text in examples:
    tokens = tokenizer.tokenize(text)
    vocabulary_set.update(tokens)

vocab_size = len(vocabulary_set)
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
encoder.save_to_file('../models/review_encoder')

encoded_examples = []
for comment in examples:
    encoded_examples.append(encoder.encode(comment))

encoded_examples = tf.keras.preprocessing.sequence.pad_sequences(encoded_examples, maxlen=50, padding='post')

shuffled_examples, shuffled_labels = shuffle_data(encoded_examples, labels)

# Split data into training and testing
x_train = shuffled_examples[:int(len(shuffled_examples) * 0.7)]
y_train = shuffled_labels[:int(len(shuffled_labels) * 0.7)]
x_test = shuffled_examples[int(len(shuffled_examples) * 0.7):]
y_test = shuffled_labels[int(len(shuffled_labels) * 0.7):]

# Convert data to a tensorflow dataset and shuffle the batches
training_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

training_data = training_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

'''
# Model structure
sentiment_analysis_model = tf.keras.Sequential()
sentiment_analysis_model.add(layers.Embedding(encoder.vocab_size, 128))
sentiment_analysis_model.add(layers.Bidirectional(tf.keras.layers.LSTM(128)))
sentiment_analysis_model.add(layers.Dense(64, activation="relu"))
sentiment_analysis_model.add(layers.Dense(1, activation="sigmoid"))
sentiment_analysis_model.summary()
'''

sentiment_analysis_model = tf.keras.Sequential()
sentiment_analysis_model.add(layers.Embedding(encoder.vocab_size, 64))
sentiment_analysis_model.add(layers.GlobalAveragePooling1D())
sentiment_analysis_model.add(layers.Dense(1))
sentiment_analysis_model.summary()


# Train and test model
sentiment_analysis_model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
history = sentiment_analysis_model.fit(training_data, epochs=3, validation_data=test_data, validation_steps=30)
sentiment_analysis_model.evaluate(test_data)

predictions = []
raw_predictions = sentiment_analysis_model.predict(x_test)
for raw_prediction in raw_predictions:
    if raw_prediction[0] > 0:
        prediction = 1
    else:
        prediction = 0
    predictions.append(prediction)

confusion_matrix = confusion_matrix(y_test, predictions)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(confusion_matrix, classes=CLASSES, title='Movie Sentiment Confusion Matrix')
plt.show()

# Save model to a file so it can be loaded later on.
tf.keras.models.save_model(sentiment_analysis_model, '../models/rotten_tomatoes_sentiment_model.h5')