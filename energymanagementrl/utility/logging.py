import logging


def get_logger(name: str):
    info_logger = logging.getLogger(f"{__name__}" + name)  # Unique logger name
    info_logger.setLevel(logging.INFO)  # Capture INFO and above

    # Ensure handler is added only once
    if not info_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)  # Ensure it logs INFO and above

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        info_logger.addHandler(handler)

    return info_logger
