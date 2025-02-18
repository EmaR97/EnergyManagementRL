import logging


def get_logger(name, lvl=logging.INFO):
    info_logger = logging.getLogger(f"{name}")
    info_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(lvl)
    console_handler.setFormatter(formatter)

    if not info_logger.handlers:
        info_logger.addHandler(console_handler)
        info_logger.addHandler(file_handler)

    return info_logger
