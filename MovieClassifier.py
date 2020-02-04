from flask import Flask, request, abort, jsonify
from flask_cors import CORS, cross_origin
import TwitterCredentials
import TweetClassifier
import PostExtractor

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

twitter_post_extractor = PostExtractor.PostExtractor(TwitterCredentials.API_KEY, TwitterCredentials.API_SECRET_KEY, TwitterCredentials.ACCESS_TOKEN, TwitterCredentials.ACCESS_TOKEN_SECRET)
tweet_classifier = TweetClassifier.TweetClassifer()

@app.route('/see-or-skip/get_sentiment_classification', methods=['POST'])
@cross_origin()
def get_movie_sentiment():
    if not request.json or not 'movie_name' in request.json:
        abort(400)

    movie_name = request.json['movie_name']
    movie_posts = twitter_post_extractor.extract_tweets(movie_name)
    sentiment_results = tweet_classifier.classify_tweet_batch(movie_posts)

    api_response = {
        "negative_comments" : sentiment_results[1],
        "neutral_comments": sentiment_results[2],
        "positive_comments": sentiment_results[3]
    }

    return jsonify(api_response), 200

if __name__ == '__main__':
    app.run(debug=True)
