#!/usr/bin/env python3

import sys, requests, json, re, urllib, random
from bs4 import BeautifulSoup
from constants import (
    GENIUS_AUTH_TOKEN,
    SPOTIFY_API_URL,
    SPOTIFY_ACCESS_TOKEN
)

#Spotify part
def request_valid_song(genre=None):
    # Wildcards for random search
    randomSongsArray = ['%25a%25', 'a%25', '%25a',
                        '%25e%25', 'e%25', '%25e',
                        '%25i%25', 'i%25', '%25i',
                        '%25o%25', 'o%25', '%25o',
                        '%25u%25', 'u%25', '%25u']
    randomSongs = random.choice(randomSongsArray)
    # Genre filter definition
    if genre:
        genreSearchString = " genre:'{}'".format(genre)
    else:
        genreSearchString = ""
    # Upper limit for random search
    maxLimit = 10000
    while True:
        try:
            randomOffset = random.randint(1, maxLimit)
            authorization_header = {
                "Authorization": "Bearer {}".format(SPOTIFY_ACCESS_TOKEN)
            }
            song_request = requests.get(
                "{}/search?query={}&offset={}&limit=1&type=track".format(
                    SPOTIFY_API_URL,
                    randomSongs + genreSearchString,
                    randomOffset
                ),
                headers=authorization_header
            )
            song_info = json.loads(song_request.text)['tracks']['items'][0]
            artist = song_info['artists'][0]['name']
            song = song_info['name']
        except IndexError:
            if maxLimit > 1000:
                maxLimit = maxLimit - 1000
            elif maxLimit <= 1000 and maxLimit > 0:
                maxLimit = maxLimit - 10
            else:
                artist = "Rick Astley"
                song = "Never gonna give you up"
                break
            continue
        break
    return [artist, song]


#Genius API part
def request_song_info(song_title, artist_name):
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + GENIUS_AUTH_TOKEN}
    search_url = base_url + '/search'
    data = {'q': song_title + ' ' + artist_name}
    response = requests.get(search_url, data=data, headers=headers)
    return response

def scrap_song_url(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')
    [h.extract() for h in html('script')]
    lyrics = html.find('div', class_='lyrics').get_text()
    return lyrics

def main():
    try:
        with open('genres.json', 'r') as infile:
            valid_genres = json.load(infile)
    except FileNotFoundError:
        print("Couldn't find genres file!")
        sys.exit(1)

    for genre in valid_genres:
        print(genre)
        with open(genre + ".txt", "w") as of:
            i = 0
            while i<10000:
                song_info = request_valid_song(genre=genre)
                artist, song = song_info
                print('{} by {}'.format(song, artist))

                # Search for matches in request response
                response = request_song_info(song, artist).json()
                remote_song_info = None

                for hit in response['response']['hits']:
                    if artist.lower() in hit['result']['primary_artist']['name'].lower():
                        remote_song_info = hit
                        break

                # Extract lyrics from URL if song was found
                if remote_song_info:
                    song_url = remote_song_info['result']['url']
                    lyrics = scrap_song_url(song_url)
                    of.write(str(i+1) + "\n")
                    of.write(song + "\t " + artist + "\n")
                    of.write("\\n".join(lyrics.strip().split('\n')) + '\n')
                    i+=1
                else:
                    print("Couldn't find")

if __name__ == '__main__':
    main()