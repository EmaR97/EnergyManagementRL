import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    raise ImportError("Please install pandas: pip install pandas")

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    raise ImportError("This script must run in a Kaggle environment with kaggle_secrets available.")

DATASET_NAME = "logs-persistence-test"
DATASET_FILE = "logs.csv"
METADATA_FILE = "dataset-metadata.json"
VERSION_MESSAGE = "Updated dataset with duplicated last row"
DELETE_OLD_VERSIONS = False

# Get Kaggle credentials
user_secrets = UserSecretsClient()
kaggle_key = user_secrets.get_secret("KAGGLE_KEY")
kaggle_username = user_secrets.get_secret("KAGGLE_USERNAME")

# Create .kaggle directory
kaggle_config_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_config_dir, exist_ok=True)

# Write kaggle.json
kaggle_json_path = os.path.join(kaggle_config_dir, 'kaggle.json')
with open(kaggle_json_path, 'w') as f:
    json.dump({"username": kaggle_username, "key": kaggle_key}, f)
os.chmod(kaggle_json_path, 0o600)

try:
    import kaggle
    from kaggle import api as kaggle_api
except ImportError:
    raise ImportError("Please install kaggle: pip install kaggle")

kaggle.api.authenticate()

DATASET_ID = f"{kaggle_username}/{DATASET_NAME}"
DATASET_DIR = f"./{DATASET_NAME}"


def download_dataset(dataset_id: str, output_dir: str, file_path: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, file_path)):
        print(f"Downloading dataset {dataset_id}...")
        kaggle_api.dataset_download_files(dataset_id, path=output_dir, unzip=True)
        kaggle_api.dataset_metadata(dataset_id, path=output_dir)
        print("Download and extraction completed.\n")


def update_metadata(file_path: str, dataset_id: str, output_dir: str):
    try:
        with open(os.path.join(output_dir, file_path), "r") as f:
            metadata = json.load(f)
        # If it's a stringified JSON, parse it; otherwise leave it as-is.
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        metadata["id"] = dataset_id
        with open(os.path.join(output_dir, file_path), "w") as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"Metadata update error: {e}")


def upload_new_version(directory: str, message: str, delete_old: bool):
    try:
        kaggle_api.dataset_create_version(directory, message, delete_old_versions=delete_old)
    except Exception as e:
        print(f"Upload error: {e}")


class KaggleDatasetHandler(logging.Handler):
    log_df: pd.DataFrame

    def emit(self, record):
        try:
            entry = {'asctime': self.format(record).split(' - ')[0], 'name': record.name, 'levelname': record.levelname,
                     'message': record.getMessage()}
            self.log_df.loc[len(self.log_df)] = entry
        except Exception as e:
            print(f"Logging error: {e}")

    def __init__(self, _log_df):
        super().__init__()
        self.log_df = _log_df


@contextmanager
def logger_to_kaggle_dataset(logger, dataset_id, dataset_dir, dataset_file, metadata_file):
    filename = os.path.join(dataset_dir, datetime.now().strftime("%Y%m%d") + "_" + dataset_file)

    handler = None  # <-- Add this
    handler_added = False  # <-- And this

    try:
        download_dataset(dataset_id, dataset_dir, metadata_file)

        if not os.path.exists(filename):
            log_df = pd.DataFrame(columns=['asctime', 'name', 'levelname', 'message'])
            log_df.to_csv(filename, index=False)
        else:
            log_df = pd.read_csv(filename)

        if not any(isinstance(h, KaggleDatasetHandler) for h in logger.handlers):
            handler = KaggleDatasetHandler(log_df)
            logger.addHandler(handler)
            handler_added = True

        yield logger
    finally:
        if handler is not None:  # <-- Important safety check
            handler.log_df.to_csv(filename, index=False)

        update_metadata(metadata_file, dataset_id, dataset_dir)
        upload_new_version(dataset_dir, VERSION_MESSAGE, DELETE_OLD_VERSIONS)

        if handler_added:
            logger.removeHandler(handler)


mylogger = logging.getLogger("TestLogger")
mylogger.setLevel(logging.INFO)

with logger_to_kaggle_dataset(mylogger, DATASET_ID, DATASET_DIR, DATASET_FILE, METADATA_FILE) as log:
    log.info(f"Notebook run at {datetime.now()}")
    log.warning("This is a warning")
    log.error("This is an error")