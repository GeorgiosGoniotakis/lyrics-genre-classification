import os
import json

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from conf.filepaths import GENRE_FILE
from lib.exceptions.GenreFileExceptions import *
from lib.data.lyrics import *


class SongsRetriever:
    GENRE_FILE_PATH = os.path.join(os.path.dirname(__file__), '../../' + GENRE_FILE)
    END_YEAR_RANGE = [2003, 2008, 2013, 2018]
    SONGS_PER_CAT = 50

    def __init__(self):
        self.spotify_cred = self.load_credentials()
        self.spotify_token = self.retrieve_token()
        self.spotify_con = self.init_connection()
        self.genres = self.load_genres()
        self.lr = LyricsRetriever()
        self.data = dict()

    def init_connection(self):
        return spotipy.Spotify(client_credentials_manager=self.spotify_token)

    def load_credentials(self):
        return Loader().spotify_params

    def retrieve_token(self):
        return SpotifyClientCredentials(client_id=self.spotify_cred["spotify_client_id"],
                                        client_secret=self.spotify_cred["spotify_client_secret"])

    def refresh_token(self):
        self.spotify_token = self.retrieve_token()
        self.spotify_con = self.init_connection()

    def load_genres(self):
        with open(self.GENRE_FILE_PATH) as f:
            return json.load(f)

    def file_exists(self):
        if not os.path.isfile(self.GENRE_FILE_PATH):
            raise GenreFileNotExists

    def search_genre(self, genre, start, end, offset):
        return self.spotify_con.search(q="genre:" + genre + " year:" + str(start) + "-" + str(end),
                                       type="track",
                                       limit=self.SONGS_PER_CAT,
                                       offset=offset)

    def generate_dataset(self):
        # For all the genres
        for g in self.genres:
            # For every year range
            for y in self.END_YEAR_RANGE:
                offset = 0
                # 10.000 Results Maximum Allowed
                while offset <= 9950:
                    res = self.search_genre(g, y - 4, y, offset)  # 5-Year Intervals
                    # If does not exist append to dictionary
                    for t in res["tracks"]["items"]:
                        if t["id"] not in self.data:
                            self.data[t["id"]] = [t["artists"][0]["name"],
                                                  t["name"],
                                                  t["album"]["release_date"],
                                                  g,
                                                  ""]
                            # For lyrics uncomment this and put it inside the arguments above
                            # lr.request_song_info(t["name"],
                            #                      t["artists"][0]).json()["response"]["hits"][0]["result"]["path"]
                    offset += 50

    def extract_dataset(self):
        df = pd.DataFrame.from_dict(self.data, orient='index')
        df.reset_index(inplace=True)
        df.columns = ["spotify_id",
                      "artist",
                      "album",
                      "year",
                      "genre",
                      "lyrics"]
        df.to_csv("../../" + DATASET_FILE, index=False)


sr = SongsRetriever()
sr.generate_dataset()
sr.extract_dataset()
