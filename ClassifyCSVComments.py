import csv
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import tensorflow as tf
import tensorflow_datasets as tfds

class ClassifyCSVComments:
    def __init__(self, model_path, encoder_path):
        self.model = tf.keras.models.load_model(model_path)
        self.encoder = tfds.features.text.TokenTextEncoder.load_from_file(encoder_path)

    # method to load comments from a csv
    def load_comments_from_csv(self, dataset_path):
        with open(dataset_path) as csvFile:
            csv_reader = csv.reader(csvFile, delimiter=',')
            comments = list(csv_reader)

        comments.pop(0)
        examples = []
        labels = []

        # add labels and examples to a list
        for comment in comments:
            examples.append(comment[0].lower())
            labels.append(int(comment[1]))

        return examples, labels

    # method to preprocess comments
    def preprocess_comments(self, examples):
        # initalise objects to preprocess
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        tokenizer = tfds.features.text.Tokenizer()

        preprocessed_examples = []
        for text in examples:
            # Tokenise
            tokens = tokenizer.tokenize(text)

            # Remove Stopwords
            filtered_tokens = []
            for token in tokens:
                if token not in stop_words:
                    filtered_tokens.append(token)

            # Stem tokens
            stemmed_tokens = []
            for token in tokens:
                stemmed_tokens.append(stemmer.stem(token))

            preprocessed_examples.append(tokenizer.join(stemmed_tokens))

        # encode comments
        encoded_examples = []
        for comment in preprocessed_examples:
            encoded_examples.append(self.encoder.encode(comment))

        # pad comments
        encoded_examples = tf.keras.preprocessing.sequence.pad_sequences(encoded_examples, maxlen=50, padding='post')

        return encoded_examples

    # method to get the comment predictions and compare to the actual labels
    def get_predictions_and_accuracy(self, examples, labels, isSoftmax):
        # get predictions
        raw_predictions = self.model.predict_on_batch(examples)

        # if its a softmax model
        if isSoftmax:
            predictions = tf.math.argmax(raw_predictions, 1)
        else:
            predictions = []
            for raw_prediction in raw_predictions:
                if raw_prediction[0] > 0:
                    predictions.append(2)
                else:
                    predictions.append(0)
        # check if the prediction is correct
        correct = 0
        for i in range(len(labels)):
            if predictions[i] == labels[i]:
                correct = correct + 1

        accuracy = correct / len(labels)

        # return prediction breakdown
        if isSoftmax:
            predictions_output = [0, 0, 0]
            for prediction in predictions:
                if prediction == 0:
                    predictions_output[0] = predictions_output[0] + 1
                elif prediction == 1:
                    predictions_output[1] = predictions_output[1] + 1
                elif prediction == 2:
                    predictions_output[2] = predictions_output[2] + 1
        else:
            predictions_output = [0, 0]
            for raw_prediction in raw_predictions:
                if raw_prediction[0] > 0:
                    predictions_output[1] = predictions_output[1] + 1
                else:
                    predictions_output[0] = predictions_output[0] + 1

        return predictions_output, accuracy

    # method to save comments to a csv file
    def save_comments_to_csv(self, file_path, comments):
        with open(file_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)

            for comment in comments:
                writer.writerow([comment, 1])
