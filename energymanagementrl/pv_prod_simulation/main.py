# main.py
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd

from .weather import get_weather_data_openmeteo, get_weather_data_clearsky
from .simulation import PlantConfig
from .simulation import run_simulation, model_config


class WeatherType(Enum):
    clear_sky = 0
    open_meteo = 1


def run_energy_production_prediction(
        plant_config: PlantConfig,
        start_time: str,
        end_time: str,
        weather_type: WeatherType = WeatherType.clear_sky,
) -> pd.DataFrame:
    mc, system, loc = model_config(plant_config)
    if weather_type == WeatherType.open_meteo:
        weather_data = get_weather_data_openmeteo(start_time, end_time, plant_config.latitude, plant_config.longitude,
                                                  plant_config.timezone)
    elif weather_type == WeatherType.clear_sky:
        weather_data = get_weather_data_clearsky(end_time, loc, plant_config.timezone, start_time)
    else:
        raise ValueError()
    return run_simulation(mc, system, weather_data[:])


def plot_results(
        results_df: pd.DataFrame
) -> None:
    plt.figure()
    results_df.plot(kind='line')
    plt.ylabel('System Output (kW)')
    plt.legend()
    plt.grid()
    plt.show()


def analyze_production(
        results_df: pd.DataFrame
):
    max_production = results_df['inverter_ac'].max()
    max_production_time = results_df['inverter_ac'].idxmax()
    production_threshold = 0
    production_start = results_df[results_df['inverter_ac'] > production_threshold].index.min()
    production_end = results_df[results_df['inverter_ac'] > production_threshold].index.max()

    print(f"Total AC energy: {results_df['inverter_ac'].sum() / 4:.2f} kWh")
    print(f"Max production: {max_production:.2f} kW at {max_production_time}")
    print(f"Production starts at: {production_start}, ends at: {production_end}")
    plot_results(results_df)

    return max_production, max_production_time, production_start, production_end
