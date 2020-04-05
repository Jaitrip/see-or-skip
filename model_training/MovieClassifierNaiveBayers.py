import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

dataset = "../data/movie_comments_dataset.csv"
CLASSES = ["Negative", "Neutral", "Positive"]

with open(dataset) as csvFile:
    csv_reader = csv.reader(csvFile, delimiter=',')
    posts = list(csv_reader)

posts.pop(0)
examples = []
labels = []

for post in posts:
    examples.append(post[0].lower())
    labels.append(int(post[1]))

x_train, x_test, y_train, y_test = train_test_split(examples, labels, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(x_train)

counts_train = vectorizer.transform(x_train)
counts_test = vectorizer.transform(x_test)

naive_bayers = MultinomialNB()
naive_bayers.fit(counts_train, y_train)
print("Training Accuracy: " + str(naive_bayers.score(counts_train, y_train)))
print("Test Accuracy: " + str(naive_bayers.score(counts_test, y_test)))