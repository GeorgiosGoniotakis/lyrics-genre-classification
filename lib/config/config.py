"""Includes the functions that load the config data into the memory."""

from pathlib import Path
import configparser

from conf.filepaths import PARAM_FILE
from lib.exceptions.ParamFileExceptions import *


class Loader:
    """"""
    CONFIG_FILE = Path("../../" + PARAM_FILE)
    PARAM_LIST = {
        "spotify": ["spotify_client_id",
                    "spotify_client_secret"],
        "genius": ["genius_auth_token"]
    }

    def __init__(self):
        self.spotify_params = None
        self.genius_params = None
        self.get_config()

    def get_config(self):
        cfg = configparser.ConfigParser()

        if self.file_exists():
            cfg.read(self.CONFIG_FILE)
            if self.args_exist(cfg):
                self.spotify_params = dict(cfg.items("spotify"))
                self.genius_params = dict(cfg.items("genius"))

    def file_exists(self):
        if not self.CONFIG_FILE.exists():
            raise NoParamFileFound(self.CONFIG_FILE)
        return True

    def args_exist(self, cfg):
        for cat in self.PARAM_LIST:
            self.cat_exists(cfg, cat)
            for item in cfg[cat]:
                self.item_exists(cfg, cat, item)
        return True

    def cat_exists(self, cfg, category):
        if category not in cfg:
            raise NoCategoryFound(category)

    def item_exists(self, cfg, category, item):
        if item not in cfg[category]:
            raise NoItemFound(item)
