from __future__ import annotations

import pandas
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from os import path
from pathlib import Path
from time import sleep, time
from typing import List


class SyncQueue:
    def __init__(self):
        self.storage = []

    def put(self, val, *args, **kwrgs):
        self.storage.append(val)

    def get(self, *args, **kwrgs):
        return self.storage.pop(0)

    def empty(self):
        return len(self.storage) == 0


class StorageController(ABC):
    def __init__(self, output_dir: Path, filenames: List[str], is_async=True, **kwrgs):
        self.output_dir = output_dir
        self.is_async = is_async
        self.queues = {}

        for key, val in kwrgs.items():
            self.__setattr__(key, val)

        if self.is_async:
            self.queues = {
                f: {"save": Queue(), "overwrite": Queue()} for f in filenames
            }

        else:
            self.queues = {
                f: {"save": SyncQueue(), "overwrite": SyncQueue()} for f in filenames
            }

        output_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        # add processes to tasks

        if self.is_async:
            for filename in self.queues:
                p = Process(
                    target=StorageController.__handle_async_queues,
                    args=(
                        self,
                        filename,
                        self.queues[filename]["save"],
                        self.queues[filename]["overwrite"],
                    ),
                )
                p.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # kill processes from tasks
        for filename in self.queues:
            self.queues[filename]["overwrite"].put((-10, None))

    def _filepath(self, filename):
        return str(self.output_dir.joinpath(filename).resolve())

    @abstractmethod
    def _write(self, filename, content):
        raise NotImplementedError

    @abstractmethod
    def _append(self, filename, content):
        raise NotImplementedError

    @abstractmethod
    def _read(self, filename):
        raise NotImplementedError

    @staticmethod
    def __handle_async_queues(
        self, filename, save_queue: Queue, overwrite_queue: Queue
    ):
        while True:
            latest_overwrite_t, latest_overwrite = -1, None
            while not overwrite_queue.empty():
                latest_overwrite_t, latest_overwrite = overwrite_queue.get()

            to_append = []

            while not save_queue.empty():
                _time_stamp, _content = save_queue.get()
                if _time_stamp > latest_overwrite_t:
                    to_append.append(_content)

            if latest_overwrite:
                self._write(filename, latest_overwrite)

            for content in to_append:
                self._append(filename, content)

            # kill signal
            if latest_overwrite_t == -10:
                return 0

    def save(self, filename, content):
        assert (
            filename in self.queues
        ), f"'{filename}' not in stream queues: {list(self.queues.keys())}"
        if self.is_async:
            self.queues[filename]["save"].put((time(), content))

        else:
            self._append(filename, content)

    def overwrite(self, filename, content):
        assert (
            filename in self.queues
        ), f"'{filename}' not in stream queues: {list(self.queues.keys())}"
        if self.is_async:
            self.queues[filename]["overwrite"].put((time(), content))

        else:
            self._write(filename, content)

    def load(self, filename):
        return self._read(filename)


class DataframeCSVStorageController(StorageController):
    def __init__(self, *args, columns={}, **kwrgs):
        super().__init__(*args, **kwrgs)
        self.columns = columns

    def _fit(self, filename, content: pandas.DataFrame | dict):
        if filename not in self.columns:
            return content

        columns = self.columns[filename]

        if isinstance(content, dict):
            _content = pandas.DataFrame([content])

        else:
            _content = content.copy()

        # add missing columns
        for col in columns:
            if col not in _content.columns:
                _content[col] = None

        return _content[columns]

    def _write(self, filename, content: pandas.DataFrame | dict):
        if len(content) == 0:
            return

        _content = self._fit(filename, _content)
        _content.to_csv(self._filepath(filename), index=False)

    def _append(self, filename, content: pandas.DataFrame):
        if len(content) == 0:
            return

        _content = self._fit(filename, content)

        mode = "w"

        if path.exists(self._filepath(filename)):
            mode = "a"

        _content.to_csv(
            self._filepath(filename), mode=mode, header=mode == "w", index=False
        )

    def _read(self, filename):
        df = pandas.read_csv(self._filepath(filename))

        return df.drop(columns=[c for c in df.columns if "Unnamed" in c])
