from doit import get_var
import yaml

import pandas as pd
import shutil
from pathlib import Path

class Config:
    def __init__(self, path=None):
        self.cfg = {}
        self.catalog_dir = None
        self.path = path
        if self.path:
            self.load(self.path)

    def ask_config_file(self):
        raise RuntimeError("GUI file dialog not available. Please specify config file path directly.")

    def load(self, path):
        # this should update the cfg instead of replacing it
        if path:
            self.catalog_dir = Path(path).parent
            with open(path, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
            for key, value in cfg.items():
                self.cfg[key] = value

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def set(self, key, value):
        self.cfg[key] = value

    def get_url(self, key):
        value = self.cfg[key]
        return Path(value.format(CATALOG_DIR=self.catalog_dir))

    def get_destination(self, file_path):
        parts = Path(file_path).name.split('_')
        country = parts[0]
        site = parts[1]
        sitecode = "_".join(parts[1:3])
        return self.geturl("output") / country / site / sitecode

    def save(self, path=None):
        path = path or self.path
        if path:
            with open(path, "w") as ymlfile:
                yaml.dump(self.cfg, ymlfile)

# single shared instance
cfg = Config(path=get_var('config', None))