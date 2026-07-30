import os

import numpy as np
import pandas as pd

from .config import START_DATE

from ..utility import get_logger

logger = get_logger(__name__)


def load_and_prepare_data(config: dict) -> pd.DataFrame:
    data_dir = config["data_paths"].get("simulation_inputs", "../data/simulation_inputs")
    num_panels = config["solar_plant"]["num_panels"]

    input_file = os.path.join(data_dir, f"complete_series.{num_panels}_panels.csv")
    df = pd.read_csv(input_file, parse_dates=["index"], index_col=["index"])
    if "GRID_VOLTAGE" in df.columns:
        df.rename(columns={"GRID_VOLTAGE": "grid_voltage"}, inplace=True)

    df["production_power_kw_altered"] = np.where(
        df["SOC"] < 100,
        df["production_power_kw"],
        df["production_power_kw_weather_dependent"],
    )

    df = df[df.index > pd.Timestamp(START_DATE)]
    logger.info(f"Loaded {len(df)} rows from {input_file}")
    return df


def build_simulation_stack(config: dict, df: pd.DataFrame):
    from ..simulation import (
        BatterySim,
        ConsumptionSim,
        GridSim,
        InverterSim,
        ProductionSimFromReal,
    )

    production_w = df.production_power_kw_altered * 1000
    production_w_weather = df.production_power_kw_weather_dependent * 1000
    optimal_w = df.production_power_kw_optimal * 1000
    consumption_w = -df.load_power_kw * 1000
    grid_voltage = df.grid_voltage

    p_sim = ProductionSimFromReal(
        power_series=production_w,
        optimal_power_series=optimal_w,
        weather_power_series=production_w_weather,
        forecast_steps=48,
    )
    c_sim = ConsumptionSim(power_series=consumption_w, daily_sample=6, forecast_steps=12)
    b_sim = BatterySim(**config["battery"])
    g_sim = GridSim(
        **{
            k: v
            for k, v in config["grid"].items()
            if k not in ("energy_price_sell_per_kwh", "energy_price_buy_per_kwh")
        },
        energy_price_sell_per_kwh=config["grid"]["energy_price_sell_per_kwh"] / 1000,
        energy_price_buy_per_kwh=config["grid"]["energy_price_buy_per_kwh"] / 1000,
        voltage_series=grid_voltage,
    )
    i_sim = InverterSim(prod_sim=p_sim, cons_sim=c_sim, batt_sim=b_sim, grid_sim=g_sim)

    return p_sim, c_sim, b_sim, g_sim, i_sim


def get_full_period(df: pd.DataFrame) -> int:
    return min(288 * 7 * 4 * 9, len(df) - 288 * 2)
