import logging


def get_logger(name):
    info_logger = logging.getLogger(f"{name}")  # Unique logger name
    info_logger.setLevel(logging.INFO)  # Capture INFO and above

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler('.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Ensure it logs INFO and above
    console_handler.setFormatter(formatter)

    # Ensure handler is added only once
    if not info_logger.handlers:
        info_logger.addHandler(console_handler)
        info_logger.addHandler(file_handler)

    return info_logger
