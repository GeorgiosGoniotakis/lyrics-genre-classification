import base64, requests, json

GENIUS_AUTH_TOKEN = '3RUsSNE0dZqqIBDxRyGF1BLWtwE9uh6mmtpytx-t5WdFPVO-cOtG_LMw7UdI4bSy'
SPOTIFY_CLIENT_ID = '09ce01991a3143c9be9d21601efa7bca'
SPOTIFY_CLIENT_SECRET = '66a3b6a500f74c72842d0a32a1870605'

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com"
API_VERSION = "v1"
SPOTIFY_API_URL = "{}/{}".format(SPOTIFY_API_BASE_URL, API_VERSION)

def get_spotify_token():
    client_token = base64.b64encode("{}:{}".format(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
                                    .encode('UTF-8')).decode('ascii')

    headers = {"Authorization": "Basic {}".format(client_token)}
    payload = {
        "grant_type": "client_credentials"
    }
    token_request = requests.post(
        SPOTIFY_TOKEN_URL, data=payload, headers=headers)
    access_token = json.loads(token_request.text)["access_token"]
    return access_token

SPOTIFY_ACCESS_TOKEN = get_spotify_token()

