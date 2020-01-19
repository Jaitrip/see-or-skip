import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

class TweetClassifer:

    def __init__(self):
        self.model = tf.keras.models.load_model('./models/twitter_model.h5')
        self.encoder = tfds.features.text.TokenTextEncoder.load_from_file('./models/encoder')

    def preprocess_batch(self, tweet_batch):
        encoded_batch = []
        for tweet in tweet_batch:
            encoded_tweet = self.encoder.encode(tweet.lower())
            encoded_batch.append(np.asarray(encoded_tweet))

        encoded_batch = np.asarray(encoded_batch)
        encoded_batch = tf.keras.preprocessing.sequence.pad_sequences(encoded_batch, maxlen=50, padding='post')
        return encoded_batch

    def classify_tweet_batch(self, tweet_batch):
        preprocessed_tweet_batch = self.preprocess_batch(tweet_batch)
        predictions = self.model.predict_on_batch(preprocessed_tweet_batch)
        predictions = tf.math.argmax(predictions, 1)
        print(predictions)
        predictions_output = [0, 0, 0, 0]

        for prediction in predictions:
            if prediction == 0:
                predictions_output[0] = predictions_output[0] + 1
            elif prediction == 1:
                predictions_output[1] = predictions_output[1] + 1
            elif prediction == 2:
                predictions_output[2] = predictions_output[2] + 1
            elif prediction == 3:
                predictions_output[3] = predictions_output[3] + 1

        return predictions_output