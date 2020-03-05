import requests

class YoutubeCommentExtractor:

    def __init__(self, api_key):
        self.api_key = api_key

    def getMovieTrailerIds(self, movie_name):
        URL = "https://www.googleapis.com/youtube/v3/search"
        PARAMS = {
            "key" : self.api_key,
            "part" : "snippet",
            "q" : movie_name,
            "maxResults" : "10"
        }

        request = requests.get(URL, PARAMS)
        response = request.json()
        movie_trailers = response["items"]
        trailer_ids = []
        for trailer in movie_trailers:
            trailer_ids.append(trailer["id"]["videoId"])

        return trailer_ids

    def getVideoComments(self, video_id):
        URL = "https://www.googleapis.com/youtube/v3/commentThreads"
        PARAMS = {
            "key" : self.api_key,
            "part" : "snippet",
            "videoId" : video_id,
            "maxResults" : "100",
        }

        request = requests.get(URL, PARAMS)
        response = request.json()
        video_comments = response["items"]
        commnents = []
        for comment in video_comments:
            commnents.append(comment["snippet"]["topLevelComment"]["snippet"]["textOriginal"])

        return commnents

    def getMovieComments(self, movie_name):
        movie_ids = self.getMovieTrailerIds(movie_name)
        all_comments = []
        for movie_id in movie_ids:
            comments = self.getVideoComments(movie_id)
            all_comments = all_comments + comments

        return all_comments



commentExtractor = YoutubeCommentExtractor("AIzaSyB5cHhVmwV8u9MOFwz8tD_FMIRf-riunW4")
print(commentExtractor.getMovieComments("Morbius"))