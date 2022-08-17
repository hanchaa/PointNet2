import os
import logging
import pathlib


def make_dir(directory):
    p = pathlib.Path(directory)
    p.mkdir(parents=True, exist_ok=True)


def get_logger(directory):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    make_dir(directory)

    file_handler = logging.FileHandler(os.path.join(directory, "log.txt"), mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
