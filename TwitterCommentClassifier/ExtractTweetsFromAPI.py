import requests
from bs4 import BeautifulSoup
from requests_oauthlib import OAuth1
import datetime
import emoji as emoji
import preprocessor as p
import re

class ExtractTweetsFromAPI:
    def __init__(self, api_key, api_secret_key, access_token, access_token_secret):
        self.auth = OAuth1(api_key, api_secret_key, access_token, access_token_secret)
        self.url = "https://api.twitter.com/1.1/search/tweets.json"

    def get_tweets_until(self, movie_name, date):
        query = movie_name + " -filter:retweets"
        params = {
            'q': query,
            'count': 100,
            'lang': 'en',
            'until': date,
            'result_type': 'mixed'
        }

        results = requests.get(url=self.url, params=params, auth=self.auth)
        tweets = results.json()
        messages = [BeautifulSoup(tweet['text'], 'html5lib').get_text() for tweet in tweets['statuses']]

        return messages

    def clean_up_tweets(self, tweets):
        p.set_options(p.OPT.URL, p.OPT.MENTION)
        tweets_text = []
        for tweet in tweets:
            clean_tweet = p.clean(tweet)
            emoji_less_tweet = emoji.demojize(clean_tweet)
            punctuation_less_tweet = re.sub('[^A-Za-z0-9 ]+', '', emoji_less_tweet)
            tweets_text.append(punctuation_less_tweet.lower())

        unique_tweets = list(set(tweets_text))

        return unique_tweets

    def extract_tweets(self, movie_name):
        week_day = datetime.datetime.now().isocalendar()[2]
        start_date = datetime.datetime.now() - datetime.timedelta(days=week_day)
        dates = [str((start_date - datetime.timedelta(days=i)).date()) for i in range(7)]

        tweets = []
        for day in dates:
            tweets = tweets + self.get_tweets_until(movie_name, day)

        clean_tweets = self.clean_up_tweets(tweets)

        return clean_tweets
