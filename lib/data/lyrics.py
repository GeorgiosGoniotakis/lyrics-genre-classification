import requests
import pandas as pd

from bs4 import BeautifulSoup

from lib.config.config import Loader
from conf.filepaths import DATASET_FILE


class LyricsRetriever:
    """"""
    GENIUS_API_URL = "https://api.genius.com"
    GENIUS_API_LYRICS = "https://genius.com"

    def __init__(self):
        self.genius_creds = self.load_credentials()

    def load_credentials(self):
        return Loader().genius_params

    def request_song_info(self, song_title, artist_name):
        headers = {'Authorization': 'Bearer ' + self.genius_creds["genius_auth_token"]}
        search_url = self.GENIUS_API_URL + '/search'
        data = {'q': song_title + ' ' + artist_name}
        response = requests.get(search_url, data=data, headers=headers)
        return response

    def scrape_song_url(self, url):
        page = requests.get(self.GENIUS_API_LYRICS + url)
        html = BeautifulSoup(page.text, 'html.parser')
        [h.extract() for h in html('script')]
        lyrics = html.find('div', class_='lyrics').get_text()
        return lyrics

    def add_lyrics(self, n_req=-1):
        """Time is calculated as 2 sec per request.

        The time to load and store the dataset is not calculated.

        Args:
            n_req: Number of requests
        """
        df = pd.read_csv("../../" + DATASET_FILE)
        df_c = df.copy()
        df_s = df_c[df.lyrics.isnull()]
        to_be_removed = list()
        last_id = 0

        print("Lyrics annotation is starting. Number of requests: {}".format(str(n_req)))

        for k, v in df_s.iterrows():

            last_id = k

            if n_req != -1 and n_req == 0:
                break

            try:
                df_c.loc[k, "lyrics"] = self.scrape_song_url(self.request_song_info(v["title"],
                                                                                    v["artist"]).json()["response"][
                                                                 "hits"][0]["result"]["path"])
            except IndexError:
                to_be_removed.append(k)
                print("Record with ID #{} was dropped.".format(str(k)))

            print("Requests left: {}".format(str(n_req)))

            n_req -= 1

        print("ID of last song annotated: {}".format(str(last_id)))
        print("Number of items without annotation: {}".format(str(df_c[df_c.lyrics.isnull()].shape[0])))
        print("Could not find lyrics for songs with IDs: {}".format(to_be_removed))
        print("Removing them from the list now.")
        df_c.drop(axis=1, index=to_be_removed, inplace=True)
        print("Records have been successfully removed.")
        print("Storing result into the dataset file...")
        df_c.to_csv("../../" + DATASET_FILE, index=False)
        print("File saved successfully!")
        print(df_c)


lr = LyricsRetriever()
lr.add_lyrics(50)  # Change this number to the number of songs you want to annotate and run this scipt
