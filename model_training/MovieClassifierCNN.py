import csv
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tensorflow import keras
from datetime import datetime
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


def shuffle_data(x_train, y_train):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    random_shuffle = np.arange(len(x_train))
    np.random.shuffle(random_shuffle)
    shuffled_x_train = x_train[random_shuffle]
    shuffled_y_train = y_train[random_shuffle]

    return shuffled_x_train, shuffled_y_train

# Data import and preprocessing
dataset = "../data/movie_comments_dataset.csv"
CLASSES = ["Negative", "Neutral", "Positive"]

tweets = []
with open(dataset) as csvFile:
    csv_reader = csv.reader(csvFile, delimiter=',')
    tweets = list(csv_reader)

tweets.pop(0)
examples = []
labels = []

for tweet in tweets:
    examples.append(tweet[0].lower())
    labels.append(int(tweet[1]))

# Tokenise, encode and pad data
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()

preprocessed_examples = []
for text in examples:
    #Tokenise
    tokens = tokenizer.tokenize(text)

    #Remove Stopwords
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)

    #Stem tokens
    stemmed_tokens = []
    for token in filtered_tokens:
        stemmed_tokens.append(stemmer.stem(token))

    preprocessed_examples.append(tokenizer.join(stemmed_tokens))
    vocabulary_set.update(stemmed_tokens)

vocab_size = len(vocabulary_set)
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
encoder.save_to_file('../models/encoder')

encoded_examples = []
for comment in preprocessed_examples:
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

BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 100

training_data = training_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

# Model structure
sentiment_analysis_model = tf.keras.Sequential()
sentiment_analysis_model.add(layers.Embedding(encoder.vocab_size, 100))
sentiment_analysis_model.add(layers.Conv1D(250, 5, activation='relu'))
sentiment_analysis_model.add(layers.GlobalMaxPooling1D())
sentiment_analysis_model.add(layers.Dropout(0.5))
sentiment_analysis_model.add(layers.Dense(3, activation='softmax'))
sentiment_analysis_model.summary()

logdir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
#run tensorboard using: python -m tensorboard.main --logdir=logs/fit

# Train and test model
sentiment_analysis_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = sentiment_analysis_model.fit(training_data, epochs=2, validation_data=test_data, validation_steps=30, callbacks=[tensorboard_callback])
sentiment_analysis_model.evaluate(test_data)

softmax_predictions = sentiment_analysis_model.predict(x_test)
predictions = []
for prediction in softmax_predictions:
    predictions.append(prediction.argmax())

confusion_matrix = confusion_matrix(y_test, predictions)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(confusion_matrix, classes=CLASSES, title='Movie Sentiment Confusion Matrix')
plt.show()

# Save model to a file so it can be loaded later on.
tf.keras.models.save_model(sentiment_analysis_model, '../models/movie_sentiment_cnn_model.h5')