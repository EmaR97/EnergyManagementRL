from typing import Any

import numpy as np
import pandas as pd

from ..lib.viz import plot_gap_fills
from ...utility import generate_gap_report, fill_gap_with_pattern, save_with_suffix, DAY, get_logger


logger = get_logger(__name__)

def _simulate_battery_through_gaps(df, config):
    """Simulate battery + grid through SOC gaps using filled external forcings.

    Dispatch order (Mode A — max self-consumption):
      1. Production and load (already filled — external forcings)
      2. Battery: charge surplus, discharge deficit (bounded by rate/capacity/SoC)
      3. Grid: voltage-based acceptance for feed-in, max_taken for import

    Sets SOC, stored_power_kw, and grid_power_kw for every NaN-SOC row.
    """
    result = df.copy()

    bat = config["battery"]
    grid_cfg = config["grid"]

    max_charge_kw = bat["max_charge_rate"] / 1000
    max_discharge_kw = bat["max_discharge_rate"] / 1000
    capacity_kwh = bat["capacity"] / 1000
    efficiency = bat.get("efficiency", 0.95)

    feed_in_max_kw = grid_cfg["feed_in_max"] / 1000
    feed_in_min_kw = grid_cfg["feed_in_min"] / 1000
    voltage_max = grid_cfg["voltage_max"]
    voltage_min = grid_cfg["voltage_min"]
    max_taken_kw = grid_cfg["max_taken_from"] / 1000

    power_per_volt = (
        (feed_in_max_kw - feed_in_min_kw) / (voltage_max - voltage_min)
        if voltage_max != voltage_min
        else 0.0
    )

    dt_h = 5 / 60

    soc = result["SOC"]
    is_nan = soc.isna()

    i = 0
    while i < len(result):
        if is_nan.iloc[i]:
            gap_start = i
            while i < len(result) and is_nan.iloc[i]:
                i += 1
            gap_end = i

            soc_kwh = soc.iloc[gap_start - 1] / 100 * capacity_kwh if gap_start > 0 else 0

            for j in range(gap_start, gap_end):
                prod_kw = result.iloc[j]["production_power_kw"]
                load_kw = result.iloc[j]["load_power_kw"]
                net_kw = prod_kw + load_kw

                # --- voltage-based grid acceptance ---
                acceptance_kw = _get_grid_acceptance(feed_in_min_kw, j, power_per_volt, result, voltage_max)

                if net_kw > 0:
                    # 1. charge battery
                    charge_kw, soc_kwh = _charge_battery(capacity_kwh, dt_h, efficiency, max_charge_kw, net_kw, soc_kwh)

                    # 2. feed remaining surplus to grid (acceptance-limited)
                    grid_feed_kw = _interact_with_grid(acceptance_kw, charge_kw, net_kw)

                    grid_kw, soc_pct, stored_kw = _compute_soc_metrics(capacity_kwh, grid_feed_kw, soc_kwh, -charge_kw)

                elif net_kw < 0:
                    # 1. discharge battery
                    deficit_kw, discharge_kw, soc_kwh = _discharge_battery(dt_h, efficiency, max_discharge_kw, net_kw,
                                                                           soc_kwh)
                    # 2. remaining deficit from grid (import-limited)
                    grid_take_kw = _interact_with_grid(max_taken_kw, discharge_kw, deficit_kw)

                    grid_kw, soc_pct, stored_kw = _compute_soc_metrics(capacity_kwh, -grid_take_kw, soc_kwh, discharge_kw)

                else:
                    grid_kw, soc_pct, stored_kw = _compute_soc_metrics(capacity_kwh, 0.0, soc_kwh, 0.0)

                result.iloc[j, result.columns.get_loc("SOC")] = soc_pct
                result.iloc[j, result.columns.get_loc("stored_power_kw")] = stored_kw
                result.iloc[j, result.columns.get_loc("grid_power_kw")] = grid_kw
        else:
            i += 1

    return result


def _compute_soc_metrics(capacity_kwh: float, grid_kw: float, soc_kwh: float, stored_kw: float) -> tuple[float, float, float]:
    soc_pct = soc_kwh / capacity_kwh * 100
    return grid_kw, soc_pct, stored_kw


def _discharge_battery(dt_h: float | int, efficiency, max_discharge_kw: Any, net_kw, soc_kwh: Any | int | float) -> \
        tuple[int | float, float | int, Any]:
    deficit_kw = -net_kw
    max_discharge_kwh = min(
        max_discharge_kw * dt_h,
        soc_kwh * efficiency,
    )
    discharge_kwh = min(deficit_kw * dt_h, max_discharge_kwh)
    discharge_kw = discharge_kwh / dt_h
    soc_kwh = max(soc_kwh - discharge_kwh / efficiency, 0)
    return deficit_kw, discharge_kw, soc_kwh


def _interact_with_grid(acceptance_kw: float | int, charge_kw: float | int, net_kw) -> float | int:
    remaining_kw = net_kw - charge_kw
    grid_feed_kw = min(remaining_kw, acceptance_kw)
    return grid_feed_kw


def _charge_battery(capacity_kwh: Any, dt_h: float | int, efficiency, max_charge_kw: Any, net_kw,
                    soc_kwh: Any | int | float) -> tuple[float | int, float | int]:
    max_charge_kwh = min(
        max_charge_kw * dt_h,
        (capacity_kwh - soc_kwh) / efficiency,
    )
    charge_kwh = min(net_kw * dt_h, max_charge_kwh)
    charge_kw = charge_kwh / dt_h
    soc_kwh = min(soc_kwh + charge_kwh * efficiency, capacity_kwh)
    return charge_kw, soc_kwh


def _get_grid_acceptance(feed_in_min_kw: Any, j: int, power_per_volt: Any | float, result, voltage_max) -> float | int:
    v = result.iloc[j].get("GRID_VOLTAGE", 240.0)
    if v >= voltage_max:
        acceptance_kw = feed_in_min_kw
    else:
        acceptance_kw = feed_in_min_kw + (voltage_max - v) * power_per_volt
    acceptance_kw = max(acceptance_kw, 0.0)
    return acceptance_kw


def run(config: dict):
    logger.info("Starting data processing and analysis")

    num_panels = config["solar_plant"]["num_panels"]
    data_dir = config["data_paths"].get("simulation_inputs", "../data/simulation_inputs")

    plant_history = pd.read_csv(
        f"{data_dir}/plant_history.csv", index_col="timestamp", parse_dates=True
    )
    forecasts = pd.read_csv(
        f"{data_dir}/forecasts.{num_panels}_panels.csv", index_col="timestamp", parse_dates=True
    )

    start = max(plant_history.index.min(), forecasts.index.min())
    end = min(plant_history.index.max(), forecasts.index.max())
    logger.info(f"Overlapping time range: {start} to {end}")
    logger.info(f"Plant history: {len(plant_history)} rows")
    logger.info(f"Forecasts: {len(forecasts)} rows")

    plant_history = plant_history.loc[start:end]
    forecasts = forecasts.loc[start:end]

    complete = pd.concat([plant_history, forecasts], axis=1)

    logger.info(f"Merged dataset: {complete.shape[0]} rows, {complete.shape[1]} columns")

    numeric_cols = complete.select_dtypes(include=[np.number]).columns
    stats = complete[numeric_cols].describe()
    logger.info("Descriptive statistics:")
    for line in stats.to_string().split("\n"):
        logger.info(f"  {line}")

    if "production_power_kw" in complete.columns:
        logger.info(f"Total production: {complete.production_power_kw.sum():.2f} kWh")
    if "load_power_kw" in complete.columns:
        logger.info(f"Total consumption: {complete.load_power_kw.abs().sum():.2f} kWh")

    generate_gap_report(complete)

    n_nan = complete.isna().sum().sum()

    date_start = start.strftime("%Y%m%d")
    date_end = end.strftime("%Y%m%d")
    suffix = f"{date_start}_{date_end}"

    if n_nan:
        logger.info(f"Filling {n_nan} NaN in merged dataset")
        before_fill = complete.copy()

        for col in ["production_power_kw", "load_power_kw", "GRID_VOLTAGE"]:
            if col in complete.columns and complete[col].isna().any():
                complete[col] = complete[col].interpolate(method="linear", limit=12)
                complete[col] = fill_gap_with_pattern(complete[col], num_days=3)

        if "SOC" in complete.columns and complete["SOC"].isna().any():
            n_soc = complete["SOC"].isna().sum()
            complete = _simulate_battery_through_gaps(complete, config)
            logger.info(f"  Simulated battery + grid through {n_soc} NaN SOC timesteps")

        plot_gap_fills(before_fill, complete, config, date_suffix=suffix)

    generate_gap_report(complete)

    save_with_suffix(
        complete, data_dir, f"complete_series.{num_panels}_panels", suffix,
        index_label="index",
    )
    logger.info(f"Processed data saved with suffix {suffix}")
