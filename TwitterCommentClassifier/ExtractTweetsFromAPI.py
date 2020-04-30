import requests
from bs4 import BeautifulSoup
from requests_oauthlib import OAuth1
import datetime
import emoji as emoji
import preprocessor as p
import re

class ExtractTweetsFromAPI:
    def __init__(self, api_key, api_secret_key, access_token, access_token_secret):
        # intialise oauth
        self.auth = OAuth1(api_key, api_secret_key, access_token, access_token_secret)
        self.url = "https://api.twitter.com/1.1/search/tweets.json"

    # get tweets between until date
    def get_tweets_until(self, movie_name, date):
        # SOURCE : https://medium.com/@jayeshsrivastava470/how-to-extract-tweets-from-twitter-in-python-47dd07f4e8e7
        # FUNCTION : Extracts 100 tweets from the twitter API
        # STATUS : changed parameters
        # BEGINS
        query = movie_name + " -filter:retweets"
        params = {
            'q': query,
            'count': 100,
            'lang': 'en',
            'until': date,
            'result_type': 'recent'
        }

        try:
            # make request to the api and save the comments in a list
            results = requests.get(url=self.url, params=params, auth=self.auth)
            tweets = results.json()
            tweets_text = [BeautifulSoup(tweet['text'], 'html5lib').get_text() for tweet in tweets['statuses']]

        # ENDS

            # return list of tweets
            return tweets_text


        except:
            print("Exception while getting tweets")
            return []

    # method to preprocess tweets
    def clean_up_tweets(self, tweets):
        # intialise preprocess object to remove urls and user mentions
        p.set_options(p.OPT.URL, p.OPT.MENTION)
        tweets_text = []
        for tweet in tweets:
            # remove urls, mentions, punctuation and emojis
            clean_tweet = p.clean(tweet)
            emoji_less_tweet = emoji.demojize(clean_tweet)
            punctuation_less_tweet = re.sub('[^A-Za-z0-9 ]+', '', emoji_less_tweet)
            tweets_text.append(punctuation_less_tweet.lower())

        # get unique tweets
        unique_tweets = list(set(tweets_text))

        return unique_tweets

    # extract tweets for the weeks
    def extract_tweets(self, movie_name):
        # get the dates of the week
        date_today = datetime.datetime.today()
        dates = [date_today - datetime.timedelta(days=i) for i in range(7)]
        dates = [dates[i].strftime("%Y-%m-%d") for i in range(7)]

        # for each date in dates get 100 tweets
        tweets = []
        for day in dates:
            dayTweets = self.get_tweets_until(movie_name, day)
            tweets = tweets + dayTweets

        # preprocess and return tweets
        clean_tweets = self.clean_up_tweets(tweets)

        return clean_tweets
