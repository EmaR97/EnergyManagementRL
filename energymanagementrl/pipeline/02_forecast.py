import os

import pandas as pd

from .config import get_plant_config

from ..utility import get_logger

logger = get_logger(__name__)


def run(config: dict):
    logger.info("Starting forecast generation")

    plant_config = get_plant_config(config)
    num_panels = config["solar_plant"]["num_panels"]
    data_dir = config["data_paths"].get("simulation_inputs", "../data/simulation_inputs")

    plant_history = pd.read_csv(
        os.path.join(data_dir, "plant_history.csv"), index_col="timestamp", parse_dates=True
    )

    start = plant_history.index.min()
    end = plant_history.index.max()

    from ..production_forecast import EnergyPredictionSystem, OpenMeteoClient, WeatherType

    open_meteo_client = OpenMeteoClient()
    energy_prediction = EnergyPredictionSystem(plant_config=plant_config, open_meteo_client=open_meteo_client)

    time_range = start.strftime("%Y-%m-%d %H:%M"), end.strftime("%Y-%m-%d %H:%M")

    open_meteo_df = energy_prediction.run_energy_production_prediction(*time_range, WeatherType.open_meteo)
    clear_sky_df = energy_prediction.run_energy_production_prediction(*time_range, WeatherType.clear_sky)

    clear_sky_df.rename({"inverter_ac": "production_power_kw_optimal"}, axis=1, inplace=True)
    open_meteo_df.rename({"inverter_ac": "production_power_kw_weather_dependent"}, axis=1, inplace=True)

    forecasts = pd.concat(
        [clear_sky_df.production_power_kw_optimal, open_meteo_df.production_power_kw_weather_dependent], axis=1
    )

    date_start = start.strftime("%Y%m%d")
    date_end = end.strftime("%Y%m%d")
    suffix = f"{date_start}_{date_end}"

    forecasts.to_csv(os.path.join(data_dir, f"forecasts.{num_panels}_panels.{suffix}.csv"), index_label="timestamp")
    forecasts.to_csv(os.path.join(data_dir, f"forecasts.{num_panels}_panels.csv"), index_label="timestamp")
    logger.info(f"Forecasts saved with suffix {suffix}")
