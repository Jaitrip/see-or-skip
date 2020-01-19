from flask import Flask, request, abort, jsonify
import TwitterCredentials
import TweetClassifier
import PostExtractor

app = Flask(__name__)

twitter_post_extractor = PostExtractor.PostExtractor(TwitterCredentials.API_KEY, TwitterCredentials.API_SECRET_KEY, TwitterCredentials.ACCESS_TOKEN, TwitterCredentials.ACCESS_TOKEN_SECRET)
tweet_classifier = TweetClassifier.TweetClassifer()

@app.route('/see-or-skip/get_sentiment_classification', methods=['POST'])
def get_movie_sentiment():
    if not request.json or not 'movie-name' in request.json:
        abort(400)

    movie_name = request.json['movie-name']
    movie_posts = twitter_post_extractor.extract_tweets(movie_name)
    sentiment_results = tweet_classifier.classify_tweet_batch(movie_posts)

    return jsonify({'movie_sentiments': sentiment_results}), 200

if __name__ == '__main__':
    app.run(debug=True)
