import ast
import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)


def run(config: dict):
    log_path = config.get("data_paths", {}).get("log_file", "../notebooks/.log")
    if not os.path.exists(log_path):
        logger.warning(f"Log file not found: {log_path}")
        return

    df = parse_log(log_path)
    if df.empty:
        logger.warning("No valid log entries found")
        return

    report = generate_report(df, config)
    logger.info(f"Monitor report:\n{report}")
    return df


def parse_log(log_path: str) -> pd.DataFrame:
    data_entries = []
    with open(log_path, "r") as f:
        for line in f:
            match = re.search(r"State: (\{.*\})", line)
            if match:
                try:
                    state_data = ast.literal_eval(match.group(1))
                    data_entries.append(state_data)
                except (ValueError, SyntaxError):
                    continue

    if not data_entries:
        return pd.DataFrame()

    df = pd.json_normalize(data_entries, sep=".")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
    return df


def generate_report(df: pd.DataFrame, config: dict = None) -> str:
    if df.empty:
        return "No data to report"

    sell_price = 0.1
    buy_price = 0.4
    if config:
        grid = config.get("grid", {})
        sell_price = grid.get("energy_price_sell_per_kwh", sell_price)
        buy_price = grid.get("energy_price_buy_per_kwh", buy_price)

    lines = [
        f"Log entries: {len(df)}",
        f"Date range: {df.index.min()} to {df.index.max()}",
    ]

    if "grid_sim.feed_to_grid" in df.columns and "grid_sim.taken_from_grid" in df.columns:
        revenue = df["grid_sim.feed_to_grid"].sum() * sell_price
        cost = df["grid_sim.taken_from_grid"].sum() * buy_price
        lines.append(f"Energy sold: {df['grid_sim.feed_to_grid'].sum():.2f} kWh")
        lines.append(f"Energy bought: {df['grid_sim.taken_from_grid'].sum():.2f} kWh")
        lines.append(f"Revenue: {revenue:.4f}")
        lines.append(f"Cost: {cost:.4f}")
        lines.append(f"Net: {revenue - cost:.4f}")

    if "action" in df.columns:
        mode_a_pct = (df["action"] == 1).mean() * 100
        lines.append(f"Mode A usage: {mode_a_pct:.1f}%")

    if "batt_sim.stored" in df.columns:
        avg_soc = df["batt_sim.stored"].mean()
        lines.append(f"Average SOC: {avg_soc:.2f}")

    return "\n".join(lines)
