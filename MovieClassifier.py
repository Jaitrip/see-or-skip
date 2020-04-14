from flask import Flask, request, abort, jsonify
from flask_cors import CORS, cross_origin
import CommentClassifier
from TwitterCommentClassifier import ExtractTweetsFromAPI, TwitterCredentials
from YoutubeCommentClassifier import YoutubeCommentExtractor
from YoutubeCommentClassifier import YoutubeCredentials

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

MODEL_PATH = "./models/movie_sentiment_cnn_model.h5"
ENCODER_PATH = "./models/encoder"

twitter_post_extractor = ExtractTweetsFromAPI.ExtractTweetsFromAPI(TwitterCredentials.API_KEY, TwitterCredentials.API_SECRET_KEY, TwitterCredentials.ACCESS_TOKEN, TwitterCredentials.ACCESS_TOKEN_SECRET)
youtube_comment_extractor = YoutubeCommentExtractor.YoutubeCommentExtractor(YoutubeCredentials.API_KEY)
classifier = CommentClassifier.CommentClassifer(model_path=MODEL_PATH, encoder_path=ENCODER_PATH)

@app.route('/see-or-skip/get_sentiment', methods=['POST'])
@cross_origin()
def get_movie_sentiment():
    if not request.json or not 'movie_name' in request.json:
        abort(400)

    movie_name = request.json['movie_name']

    movie_tweets = twitter_post_extractor.extract_tweets(movie_name)
    tweet_sentiment_results = classifier.classify_comments(movie_tweets)
    print(tweet_sentiment_results)

    movie_youtube_comments = youtube_comment_extractor.getMovieComments(movie_name)
    comment_sentiment_results = classifier.classify_comments(movie_youtube_comments)
    print(comment_sentiment_results)

    api_response = {
        "twitter_negative_comments": tweet_sentiment_results[0],
        "twitter_neutral_comments": tweet_sentiment_results[1],
        "twitter_positive_comments": tweet_sentiment_results[2],
        "youtube_negative_comments": comment_sentiment_results[0],
        "youtube_neutral_comments": comment_sentiment_results[1],
        "youtube_positive_comments": comment_sentiment_results[2]
    }

    return jsonify(api_response), 200

if __name__ == '__main__':
    app.run(debug=True)
