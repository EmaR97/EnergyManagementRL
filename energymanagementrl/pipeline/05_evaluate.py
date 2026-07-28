import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run(config: dict):
    logger.info("Starting model evaluation")

    data_dir = config["data_paths"].get("simulation_inputs", "../data/simulation_inputs")
    logs_dir = config["data_paths"].get("logs", "../data/logs")
    trained_dir = config["data_paths"].get("trained_models", "../data/trained_models")
    num_panels = config["solar_plant"]["num_panels"]

    input_file = os.path.join(data_dir, f"complete_series.{num_panels}panels.csv")
    df = pd.read_csv(input_file, parse_dates=["index"], index_col=["index"])
    if "GRID_VOLTAGE" in df.columns:
        df.rename(columns={"GRID_VOLTAGE": "grid_voltage"}, inplace=True)

    df["production_power_kw_altered"] = np.where(
        df["SOC"] < 100, df["production_power_kw"], df["production_power_kw_weather_dependent"]
    )
    df = df[df.index > pd.Timestamp("2024-10-19")]

    production_w = df.production_power_kw_altered * 1000
    production_w_weather = df.production_power_kw_weather_dependent * 1000
    optimal_w = df.production_power_kw_optimal * 1000
    consumption_w = -df.load_power_kw * 1000
    grid_voltage = df.grid_voltage

    from ..simulation import (
        ProductionSimFromReal,
        ConsumptionSim,
        BatterySim,
        GridSim,
        InverterSim,
    )
    from ..rl.env import InverterEnv
    from ..rl.models import GreedyModel, ConservativeModel

    p_sim = ProductionSimFromReal(
        power_series=production_w,
        optimal_power_series=optimal_w,
        weather_power_series=production_w_weather,
        forecast_steps=48,
    )
    c_sim = ConsumptionSim(power_series=consumption_w, daily_sample=6, forecast_steps=12)
    b_sim = BatterySim(**config["battery"])
    g_sim = GridSim(**config["grid"], voltage_series=grid_voltage)
    i_sim = InverterSim(prod_sim=p_sim, cons_sim=c_sim, batt_sim=b_sim, grid_sim=g_sim)

    full_period = min(288 * 7 * 4 * 9, len(df) - 288 * 2)
    env = InverterEnv(i_sim, full_period)

    results = {}
    for name, model_fn in [
        ("greedy", lambda: GreedyModel()),
        ("conservative", lambda: ConservativeModel()),
    ]:
        model = model_fn()
        score = _evaluate_model(env, model, full_period, name)
        results[name] = score

    best_path = os.path.join(logs_dir, "best_model", "best_model")
    if os.path.exists(best_path):
        from stable_baselines3 import DQN

        model = DQN.load(best_path, env=env)
        results["best"] = _evaluate_model(env, model, full_period, "best")

    current_model_path = config.get("models", {}).get("current_used")
    if current_model_path:
        full_path = os.path.join(trained_dir, "models", current_model_path)
        if os.path.exists(full_path):
            from ..rl.utils import load_model_with_weights

            model = load_model_with_weights(env, full_path)
            results["current_used"] = _evaluate_model(env, model, full_period, "current_used")

    results_df = pd.DataFrame([{"model": k, "score": v} for k, v in results.items()])
    results_path = os.path.join(trained_dir, "evaluation_results.csv")
    os.makedirs(trained_dir, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Results:\n{results_df.to_string()}")


def _evaluate_model(env, model, max_steps, name):
    from ..rl.utils import test_plot

    env.set_max_steps(max_steps)
    score = test_plot(env, model, max_steps, to_show=["batt_sim.stored"], seed=33)
    logger.info(f"{name}: {score}")
    return score
