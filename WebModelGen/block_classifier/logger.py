"""Logging utility"""

import logging
from tqdm import tqdm

LOGGER_NAME = "webgraph"
LOGGER_FORMAT = "%(levelname)s %(message)s"
LOGGER_LEVEL = logging.DEBUG

class TqdmStream:
    def __init__(self) -> None:
        pass

    def write(self, message):
        tqdm.write(message)

    def flush(self):
        pass

def configure_logger() -> logging.Logger:
    """Configure the logger used by Webgraph."""
    formatter = logging.Formatter(LOGGER_FORMAT)
    handler = logging.StreamHandler(stream=TqdmStream())
    handler.setLevel(LOGGER_LEVEL)
    handler.setFormatter(formatter)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LOGGER_LEVEL)
    logger.addHandler(handler)

    return logger


LOGGER = configure_logger()
