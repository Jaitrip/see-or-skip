import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import tensorflow as tf
import tensorflow_datasets as tfds

class CommentClassifer:

    def __init__(self, model_path, encoder_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.tokenizer = tfds.features.text.Tokenizer()
        self.encoder = tfds.features.text.TokenTextEncoder.load_from_file(encoder_path)

    # preprocess comments
    def preprocess_comments(self, comments):
        preprocessed_comments = []
        for text in comments:
            # Tokenise
            tokens = self.tokenizer.tokenize(text)

            # Remove Stopwords
            filtered_tokens = []
            for token in tokens:
                if token not in self.stop_words:
                    filtered_tokens.append(token)

            # Stem tokens
            stemmed_tokens = []
            for token in filtered_tokens:
                stemmed_tokens.append(self.stemmer.stem(token))

            preprocessed_comments.append(self.tokenizer.join(stemmed_tokens))

        # encode the comments
        encoded_comments = []
        for comment in preprocessed_comments:
            encoded_comments.append(self.encoder.encode(comment))

        # pad the comments
        encoded_examples = tf.keras.preprocessing.sequence.pad_sequences(encoded_comments, maxlen=50, padding='post')
        return encoded_examples

    # classify comments
    def classify_comments(self, comments):
        # use model to make predictions on each tweet
        preprocessed_comments = self.preprocess_comments(comments)
        predictions = self.model.predict_on_batch(preprocessed_comments)
        predictions = tf.math.argmax(predictions, 1)
        print(predictions)
        predictions_output = [0, 0, 0]

        # create an array of containing the counts for each category
        for prediction in predictions:
            if prediction == 0:
                predictions_output[0] = predictions_output[0] + 1
            elif prediction == 1:
                predictions_output[1] = predictions_output[1] + 1
            elif prediction == 2:
                predictions_output[2] = predictions_output[2] + 1

        return predictions_output