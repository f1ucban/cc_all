import logging
import os


def setup_logger(name):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler("/home/lucban/cc_all/src/utils/activity.log"), logging.StreamHandler()],
    )

    return logging.getLogger(name)
