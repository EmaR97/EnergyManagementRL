import logging
import os
from datetime import datetime

import pandas as pd

from energymanagementrl.utility.utility_kaggle_dataset import download_dataset, upload_new_version, update_metadata


VERSION_MESSAGE = "Updated dataset with duplicated last row"
DELETE_OLD_VERSIONS = False
class LoggerToKaggleDataset:
    def __init__(self, logger, dataset_id, dataset_dir, dataset_file, metadata_file):
        self.logger = logger
        self.dataset_id = dataset_id
        self.dataset_dir = dataset_dir
        self.dataset_file = dataset_file
        self.metadata_file = metadata_file
        self.filename = os.path.join(dataset_dir, datetime.now().strftime("%Y%m%d") + "_" + dataset_file)
        self.handler = None
        self.handler_added = False

    def __enter__(self):
        # Download dataset and prepare logging dataframe if not exists
        download_dataset(self.dataset_id, self.dataset_dir, self.metadata_file)

        if not os.path.exists(self.filename):
            log_df = pd.DataFrame(columns=['asctime', 'name', 'levelname', 'message'])
            log_df.to_csv(self.filename, index=False)
        else:
            log_df = pd.read_csv(self.filename)

        # Add a KaggleDatasetHandler to the logger if not already present
        if not any(isinstance(h, KaggleDatasetHandler) for h in self.logger.handlers):
            self.handler = KaggleDatasetHandler(log_df)
            self.logger.addHandler(self.handler)
            self.handler_added = True

        return self.logger

    def __exit__(self, exc_type, exc_value, traceback):
        # Save the log dataframe if handler was added
        if self.handler is not None:  # Safety check
            self.handler.log_df.to_csv(self.filename, index=False)

        # Update metadata and upload the new version
        update_metadata(self.metadata_file, self.dataset_id, self.dataset_dir)
        upload_new_version(self.dataset_dir, VERSION_MESSAGE, DELETE_OLD_VERSIONS)

        # Remove the handler if it was added
        if self.handler_added:
            self.logger.removeHandler(self.handler)


class KaggleDatasetHandler(logging.Handler):
    log_df: pd.DataFrame

    def emit(self, record):
        try:
            entry = {'asctime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'name': record.name,
                'levelname': record.levelname, 'message': record.getMessage()}
            self.log_df.loc[len(self.log_df)] = entry
        except Exception as e:
            print(f"Logging error: {e}")

    def __init__(self, _log_df):
        super().__init__()
        self.log_df = _log_df
