import json
from os import path
from typing import List
from .storage import StorageController


class JSONStorageController(StorageController):
    def _write(self, filename, content: List[dict]):

        if not isinstance(content, list):
            raise NotImplementedError(f"Not implemented for {type(content)}")

        with open(self._filepath(filename), "w") as f:
            json.dump(content, f)

    def _append(self, filename, content: List[dict]):
        if self.json_mode == "list":

            ls = []

            if path.exists(self._filepath(filename)):
                with open(self._filepath(filename), "r") as f:
                    ls = json.load(f)

            ls.extend(content)
            with open(self._filepath(filename), "w") as f:
                json.dump(ls, f)

        else:
            raise NotImplementedError(f"Not implemented for {self.json_mode}")

    def _read(self, filename):
        ls = []
        with open(self._filepath(filename), "r") as f:
            ls = json.load(f)

        return ls
