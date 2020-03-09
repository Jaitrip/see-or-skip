import tweepy as tweepy
import emoji as emoji
import preprocessor as p
import TwitterCredentials
import csv
import re

class PostExtractor:

    def __init__(self, api_key, api_secret_key, access_token, access_token_secret):
        self.api_key = api_key
        self.api_secret_key = api_secret_key
        self.access_token = access_token
        self.access_token_secret = access_token_secret

    def clean_up_tweets(self, tweets):
        p.set_options(p.OPT.URL, p.OPT.MENTION)
        tweets_text = []
        for tweet in tweets:
            if not tweet.retweeted and 'RT' not in tweet.full_text:
                print(tweet.full_text)
                clean_tweet = p.clean(tweet.full_text)
                emoji_less_tweet = emoji.demojize(clean_tweet)
                punctuation_less_tweet = re.sub('[^A-Za-z0-9 ]+', '', emoji_less_tweet)
                tweets_text.append(punctuation_less_tweet)

        # Get only unique tweets
        unique_tweets = list(set(tweets_text))

        return unique_tweets

    def extract_tweets(self, movie_name):
        auth = tweepy.OAuthHandler(self.api_key, self.api_secret_key)
        auth.set_access_token(self.access_token, self.access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)

        tweets = tweepy.Cursor(api.search, q=movie_name, lang='en', tweet_mode='extended').items(100)
        print("tweets extracted")
        unique_clean_tweets = self.clean_up_tweets(tweets)
        print("tweets cleaned")
        return unique_clean_tweets

    def extract_training_data(self, movie_name, dataset_path):
        tweets = self.extract_tweets(movie_name)

        # Save tweets to a CSV file
        with open(dataset_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)

            for tweet in tweets:
                writer.writerow([tweet, 2])


#post_extractor = PostExtractor(TwitterCredentials.API_KEY, TwitterCredentials.API_SECRET_KEY, TwitterCredentials.ACCESS_TOKEN, TwitterCredentials.ACCESS_TOKEN_SECRET)
#post_extractor.extract_training_data("Black Widow", "./data/twitter_data.csv")