import json
import os

import kaggle
from kaggle import api as kaggle_api


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
