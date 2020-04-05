import requests
import csv
import emoji as emoji
import re

class YoutubeCommentExtractor:

    def __init__(self, api_key):
        self.api_key = api_key

    def getMovieTrailerIds(self, movie_name):
        URL = "https://www.googleapis.com/youtube/v3/search"
        PARAMS = {
            "key" : self.api_key,
            "part" : "snippet",
            "q" : movie_name + " trailer",
            "type" : "video",
            "maxResults" : "10"
        }

        try:
            request = requests.get(URL, PARAMS)
            response = request.json()
            movie_trailers = response["items"]
            trailer_ids = []
            for trailer in movie_trailers:
                trailer_ids.append(trailer["id"]["videoId"])

            return trailer_ids

        except:
            print("Exception getting trailer ids")
            return []

    def getVideoComments(self, video_id):
        URL = "https://www.googleapis.com/youtube/v3/commentThreads"
        PARAMS = {
            "key" : self.api_key,
            "part" : "snippet",
            "videoId" : video_id,
            "maxResults" : "100",
        }

        try :
            request = requests.get(URL, PARAMS)
            response = request.json()
            video_comments = response["items"]
            comments = []
            for comment in video_comments:
                comments.append(comment["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
                if int(comment["snippet"]["totalReplyCount"]) > 10:
                    replies = self.getCommentReplies(comment["snippet"]["topLevelComment"]["id"])
                    comments = comments + replies

            return comments

        except:
            print("Exception getting comments")
            return []

    def getCommentReplies(self, comment_id):
        URL = "https://www.googleapis.com/youtube/v3/comments"
        PARAMS = {
            "part" : "snippet",
            "parentId" : comment_id,
            "maxResults" : "100",
            "key": self.api_key
        }

        try:
            request = requests.get(URL, PARAMS)
            response = request.json()
            comment_responses = response["items"]
            comments_text = []
            for response in comment_responses:
                comments_text.append(response["snippet"]["textOriginal"])

            return comments_text

        except:
            print("Exception getting comment replies")
            return []

    def getMovieComments(self, movie_name):
        movie_ids = self.getMovieTrailerIds(movie_name)
        all_comments = []
        for movie_id in movie_ids:
            comments = self.getVideoComments(movie_id)
            all_comments = all_comments + comments

        clean_comments = self.clean_comments(all_comments)
        return clean_comments


    def clean_comments(self, comments):
        preprocessed_comments = []
        for comment in comments:
            emoji_less_comment = emoji.demojize(comment)
            punctuation_less_comment = re.sub('[^A-Za-z0-9 ]+', '', emoji_less_comment)
            preprocessed_comments.append(punctuation_less_comment)

        return preprocessed_comments


    def saveCommentsToCSV(self, movie_name, dataset_path):
        comments = self.getMovieComments(movie_name)
        with open(dataset_path, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)

            for comment in comments:
                writer.writerow([comment, 2])

#extractor = YoutubeCommentExtractor("AIzaSyB5cHhVmwV8u9MOFwz8tD_FMIRf-riunW4")
#extractor.saveCommentsToCSV("Ghostbusters Trailer", "../data/compare_data.csv")