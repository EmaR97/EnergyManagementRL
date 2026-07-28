import logging
import os

import numpy as np
import pandas as pd

from .config import get_env

logger = logging.getLogger(__name__)


def run(config: dict):
    logger.info("Starting data ingestion from FusionSolar API")

    from ..fusion_solar_connector import FusionSolarClientParsed

    username = get_env("FUSION_SOLAR_CLIENT_USERNAME", required=True)
    password = get_env("FUSION_SOLAR_CLIENT_PASSWORD", required=True)
    plant_cfg = config["solar_plant"]

    client = FusionSolarClientParsed(
        username, password, huawei_subdomain=plant_cfg["inverter"]["huawei_subdomain"]
    )
    plant_id = client.get_plant_ids()[0]
    battery_id = client.get_battery_ids(plant_id)[0]
    inverter_id = plant_cfg["inverter"]["id"]
    timezone = plant_cfg["timezone"]

    current_time = pd.Timestamp.now()
    grid_connection_time = pd.to_datetime(
        client.get_plant_details(plant_id)["gridConnectedTime"].split()[0]
    )

    raw_history, final_history = client.get_plant_history(
        grid_connection_time, current_time, battery_id, inverter_id, plant_id, timezone
    )

    data_dir = config["data_paths"].get("simulation_inputs", "../data/simulation_inputs")
    os.makedirs(data_dir, exist_ok=True)

    final_clean = final_history.dropna()
    raw_clean = raw_history.loc[final_clean.index]

    date_start = final_clean.index.min().strftime("%Y%m%d")
    date_end = final_clean.index.max().strftime("%Y%m%d")
    suffix = f"{date_start}_{date_end}"

    raw_clean.replace([np.inf, -np.inf], np.nan).to_csv(
        os.path.join(data_dir, f"plant_history_raw.{suffix}.csv"), index_label="timestamp"
    )
    final_clean.replace([np.inf, -np.inf], np.nan).to_csv(
        os.path.join(data_dir, f"plant_history.{suffix}.csv"), index_label="timestamp"
    )
    raw_clean.replace([np.inf, -np.inf], np.nan).to_csv(
        os.path.join(data_dir, "plant_history_raw.csv"), index_label="timestamp"
    )
    final_clean.replace([np.inf, -np.inf], np.nan).to_csv(
        os.path.join(data_dir, "plant_history.csv"), index_label="timestamp"
    )
    logger.info(f"Ingested data saved with suffix {suffix}")


def ingest_fusion_solar(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """High-level interface for programmatic use."""
    from ..fusion_solar_connector import FusionSolarClientParsed

    username = get_env("FUSION_SOLAR_CLIENT_USERNAME", required=True)
    password = get_env("FUSION_SOLAR_CLIENT_PASSWORD", required=True)
    plant_cfg = config["solar_plant"]

    client = FusionSolarClientParsed(
        username, password, huawei_subdomain=plant_cfg["inverter"]["huawei_subdomain"]
    )
    plant_id = client.get_plant_ids()[0]
    battery_id = client.get_battery_ids(plant_id)[0]
    inverter_id = plant_cfg["inverter"]["id"]
    timezone = plant_cfg["timezone"]

    current_time = pd.Timestamp.now()
    grid_connection_time = pd.to_datetime(
        client.get_plant_details(plant_id)["gridConnectedTime"].split()[0]
    )

    return client.get_plant_history(
        grid_connection_time, current_time, battery_id, inverter_id, plant_id, timezone
    )

