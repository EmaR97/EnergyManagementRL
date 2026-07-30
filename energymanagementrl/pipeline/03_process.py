import os
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from ..utility import get_logger

logger = get_logger(__name__)

DAY = 288  # timesteps per day (5-min intervals)


def _fill_gap_with_pattern(col: pd.Series, num_days: int = 3) -> pd.Series:
    """Replace full days containing any NaN with the mean of the same step from clean days.

    For each day that has one or more NaN in the original data, the **entire** day
    (all 288 timesteps) is reconstructed from clean days before and after the gap.
    This avoids jumps when transitioning from real to filled data mid-day.
    """
    n = len(col)
    gap_days: set[int] = set()
    for i in range(n):
        if pd.isna(col.iloc[i]):
            gap_days.add(i // DAY)

    if not gap_days:
        return col.copy()

    result = col.copy()
    for day_idx in sorted(gap_days):
        day_start = day_idx * DAY
        day_end = min(day_start + DAY, n)

        for j in range(day_start, day_end):
            step = j % DAY

            pre = []
            k = j - DAY
            while len(pre) < num_days and k >= 0:
                if (k // DAY) not in gap_days and not pd.isna(col.iloc[k]):
                    pre.append(col.iloc[k])
                k -= DAY

            post = []
            k = j + DAY
            while len(post) < num_days and k < n:
                if (k // DAY) not in gap_days and not pd.isna(col.iloc[k]):
                    post.append(col.iloc[k])
                k += DAY

            vals = pre + post
            if vals:
                result.iloc[j] = np.mean(vals)

    return result


def _gap_report(df: pd.DataFrame):
    """Log a detailed gap analysis deduplicated across columns."""
    n_nan_total = df.isna().sum().sum()
    if n_nan_total == 0:
        logger.info("No missing values found")
        return

    logger.info(f"Total NaN cells: {n_nan_total}")
    for col in df.columns:
        n = df[col].isna().sum()
        if n:
            pct = n / len(df) * 100
            logger.info(f"  {col}: {n} NaN ({pct:.2f}%)")

    any_nan = df.isna().any(axis=1)
    gaps = []
    start = None
    for i, is_nan in enumerate(any_nan):
        if is_nan and start is None:
            start = i
        elif not is_nan and start is not None:
            gaps.append((df.index[start], df.index[i - 1], i - start))
            start = None
    if start is not None:
        gaps.append((df.index[start], df.index[-1], len(df) - start))

    logger.info(f"Total gap blocks: {len(gaps)}")
    if not gaps:
        return

    dist = Counter(g[2] for g in gaps)
    logger.info("Gap size distribution:")
    for size, count in sorted(dist.items(), key=lambda x: -x[1])[:10]:
        days = size / DAY
        logger.info(f"  {size:>5} rows ({days:.2f} days): {count} occurrences")

    logger.info("10 largest gaps:")
    for frm, to, length in sorted(gaps, key=lambda g: -g[2])[:10]:
        days = length / DAY
        logger.info(f"  {frm:%Y-%m-%d %H:%M}  →  {to:%Y-%m-%d %H:%M}  ({length} rows, {days:.2f} days)")


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

                    grid_kw, soc_pct, stored_kw = method_name(capacity_kwh, grid_feed_kw, soc_kwh, -charge_kw)

                elif net_kw < 0:
                    # 1. discharge battery
                    deficit_kw, discharge_kw, soc_kwh = _discharge_battery(dt_h, efficiency, max_discharge_kw, net_kw,
                                                                           soc_kwh)
                    # 2. remaining deficit from grid (import-limited)
                    grid_take_kw = _interact_with_grid(max_taken_kw, discharge_kw, deficit_kw)

                    grid_kw, soc_pct, stored_kw = method_name(capacity_kwh, -grid_take_kw, soc_kwh, discharge_kw)

                else:
                    grid_kw, soc_pct, stored_kw = method_name(capacity_kwh, 0.0, soc_kwh, 0.0)

                result.iloc[j, result.columns.get_loc("SOC")] = soc_pct
                result.iloc[j, result.columns.get_loc("stored_power_kw")] = stored_kw
                result.iloc[j, result.columns.get_loc("grid_power_kw")] = grid_kw
        else:
            i += 1

    return result


def method_name(capacity_kwh: Any, grid_kw: float, soc_kwh: Any | int | float | Any, stored_kw: float) -> \
        tuple[Any, float, float]:
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


def _plot_gaps(before: pd.DataFrame, after: pd.DataFrame, config: dict, date_suffix: str = ""):
    """Plot each large gap with a ±3-day window, comparing before/after fill."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(
        config["data_paths"].get("simulation_inputs", "../data/simulation_inputs"), "gap_plots"
    )
    if date_suffix:
        plots_dir = os.path.join(plots_dir, date_suffix)
    os.makedirs(plots_dir, exist_ok=True)

    any_nan = before.isna().any(axis=1)
    cols = ["production_power_kw", "load_power_kw", "GRID_VOLTAGE", "SOC", "stored_power_kw", "grid_power_kw"]
    titles = ["Production (kW)", "Load (kW)", "Voltage (V)", "SOC (%)", "Stored (kW)", "Grid (kW)"]
    window_days = 3
    width = max(1, len(str(len(before))))

    pos = 0
    n = len(before)
    while pos < n:
        if not any_nan.iloc[pos]:
            pos += 1
            continue

        gap_start = pos
        while pos < n and any_nan.iloc[pos]:
            pos += 1
        gap_end = pos

        if gap_end - gap_start <= 1:
            continue

        w0 = max(0, gap_start - DAY * window_days)
        w1 = min(n, gap_end + DAY * window_days)
        idx = before.index[w0:w1]

        fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
        fig.suptitle(
            f"Gap  {before.index[gap_start]:%Y-%m-%d %H:%M}  →  {before.index[min(gap_end, n) - 1]:%Y-%m-%d %H:%M}  ({gap_end - gap_start} rows)",
            fontsize=13,
        )

        for (ax_grp, col, title) in zip(axes.flatten(), cols, titles):
            if col not in before.columns:
                ax_grp.set_visible(False)
                continue
            ax_grp.plot(
                idx, before[col].iloc[w0:w1].values,
                color="#e74c3c", alpha=0.5, linewidth=0.8, label="Before fill",
            )
            ax_grp.plot(
                idx, after[col].iloc[w0:w1].values,
                color="#2980b9", alpha=0.8, linewidth=1, label="After fill",
            )
            ax_grp.axvspan(before.index[gap_start], before.index[min(gap_end, n) - 1],
                           alpha=0.12, color="gray")
            ax_grp.set_ylabel(title, fontsize=9)
            ax_grp.legend(fontsize=7, loc="upper right")
            ax_grp.grid(True, alpha=0.25)

        axes[-1, -1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        plt.tight_layout()

        fname = (
            f"{gap_end - gap_start:0{width}d}_"
            f"gap_{before.index[gap_start]:%Y%m%d_%H%M}_"
            f"{before.index[min(gap_end, n) - 1]:%Y%m%d_%H%M}.png"
        )
        fig.savefig(os.path.join(plots_dir, fname), dpi=150)
        plt.close(fig)

    n_plots = len([f for f in os.listdir(plots_dir) if f.endswith(".png")])
    logger.info(f"Saved {n_plots} gap-plot(s) to {plots_dir}")


def run(config: dict):
    logger.info("Starting data processing and analysis")

    num_panels = config["solar_plant"]["num_panels"]
    data_dir = config["data_paths"].get("simulation_inputs", "../data/simulation_inputs")

    plant_history = pd.read_csv(
        os.path.join(data_dir, "plant_history.csv"), index_col="timestamp", parse_dates=True
    )
    forecasts = pd.read_csv(
        os.path.join(data_dir, f"forecasts.{num_panels}_panels.csv"), index_col="timestamp", parse_dates=True
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

    _gap_report(complete)

    n_nan = complete.isna().sum().sum()

    date_start = start.strftime("%Y%m%d")
    date_end = end.strftime("%Y%m%d")
    suffix = f"{date_start}_{date_end}"

    if n_nan:
        logger.info(f"Filling {n_nan} NaN in merged dataset")
        before_fill = complete.copy()

        # Phase 1: fill small gaps with interpolation, then replace full days containing any remaining NaN
        for col in ["production_power_kw", "load_power_kw", "GRID_VOLTAGE"]:
            if col in complete.columns and complete[col].isna().any():
                complete[col] = complete[col].interpolate(method="linear", limit=12)
                complete[col] = _fill_gap_with_pattern(complete[col], num_days=3)

        # Phase 2: simulate battery + grid through gaps using filled forcings
        if "SOC" in complete.columns and complete["SOC"].isna().any():
            n_soc = complete["SOC"].isna().sum()
            complete = _simulate_battery_through_gaps(complete, config)
            logger.info(f"  Simulated battery + grid through {n_soc} NaN SOC timesteps")

        _plot_gaps(before_fill, complete, config, date_suffix=suffix)

    _gap_report(complete)

    complete.to_csv(os.path.join(data_dir, f"complete_series.{num_panels}_panels.{suffix}.csv"), index_label="index")
    complete.to_csv(os.path.join(data_dir, f"complete_series.{num_panels}_panels.csv"), index_label="index")
    logger.info(f"Processed data saved with suffix {suffix}")
