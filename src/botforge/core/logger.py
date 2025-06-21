import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str = "botforge") -> logging.Logger:
    log_dir = "/opt/logs/botforge"
    os.makedirs(log_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{name}_{date_str}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

log = setup_logger()
