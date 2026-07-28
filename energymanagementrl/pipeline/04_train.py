import copy
import logging
import os
import time

import numpy as np
import pandas as pd

START_DATE = "2024-10-19"

logger = logging.getLogger(__name__)


def run(config: dict):
    logger.info("Starting model training")

    train_cfg = config["training"]

    if train_cfg["device"] == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "Config requests device='cuda' but CUDA is not available. "
                    "Training will fall back to CPU. "
                    "Set training.device to 'cpu' in config.json to suppress this warning."
                )
        except ImportError:
            logger.warning("PyTorch not found — cannot verify CUDA availability.")

    logger.info(f"Using device: {train_cfg['device']}")

    data_dir = config["data_paths"].get("simulation_inputs", "../data/simulation_inputs")
    logs_dir = config["data_paths"].get("logs", "../data/logs")
    num_panels = config["solar_plant"]["num_panels"]

    os.makedirs(logs_dir, exist_ok=True)

    logger.info("Loading complete series CSV")
    input_file = os.path.join(data_dir, f"complete_series.{num_panels}_panels.csv")
    df = pd.read_csv(input_file, parse_dates=["index"], index_col=["index"])
    if "GRID_VOLTAGE" in df.columns:
        df.rename(columns={"GRID_VOLTAGE": "grid_voltage"}, inplace=True)

    logger.info(f"Loaded {len(df)} rows. Preparing data series")
    df["production_power_kw_altered"] = np.where(
        df["SOC"] < 100, df["production_power_kw"], df["production_power_kw_weather_dependent"]
    )
    # df = df[df.index > pd.Timestamp(START_DATE)]
    df = pd.concat([df, df[-288 * 2:]])

    production_w = df.production_power_kw_altered * 1000
    production_w_weather = df.production_power_kw_weather_dependent * 1000
    optimal_w = df.production_power_kw_optimal * 1000
    consumption_w = -df.load_power_kw * 1000
    grid_voltage = df.grid_voltage

    full_period = min(288 * 7 * 4 * 9, len(df) - 288 * 2)

    from ..simulation import (
        ProductionSimFromReal,
        ConsumptionSim,
        BatterySim,
        GridSim,
        InverterSim,
        week,
    )
    from ..rl.env import InverterEnvBatteryMgmt

    logger.info("Building simulation stack")
    p_sim = ProductionSimFromReal(
        power_series=production_w,
        optimal_power_series=optimal_w,
        weather_power_series=production_w_weather,
        forecast_steps=48,
    )
    c_sim = ConsumptionSim(power_series=consumption_w, daily_sample=6, forecast_steps=12)
    b_sim = BatterySim(**config["battery"])
    g_sim = GridSim(
        **{k: v for k, v in config["grid"].items() if k != "energy_price_sell_per_kwh" and k != "energy_price_buy_per_kwh"},
        energy_price_sell_per_kwh=config["grid"]["energy_price_sell_per_kwh"] / 1000,
        energy_price_buy_per_kwh=config["grid"]["energy_price_buy_per_kwh"] / 1000,
        voltage_series=grid_voltage,
    )
    i_sim = InverterSim(prod_sim=p_sim, cons_sim=c_sim, batt_sim=b_sim, grid_sim=g_sim)

    logger.info("Configuring reward and reserve parameters")
    rewards = train_cfg["rewards"]
    reserves = train_cfg["battery_reserves"]

    logger.info("Creating training environment")
    train_env = InverterEnvBatteryMgmt(
        inverter_sim=copy.deepcopy(i_sim),
        max_steps=week * 2,
        reward_near_full=rewards["reward_near_full"],
        penalty_below_night_reserve=rewards["penalty_below_night_reserve"],
        penalty_below_min_reserve=rewards["penalty_below_min_reserve"],
        min_reserve=reserves["min_reserve"],
        night_reserver=reserves["night_reserver"],
        near_full=reserves["near_full"],
    )
    train_env.shuffle = train_cfg["shuffle"]
    train_env.inverter_sim.grid_sim.energy_price_buy = rewards["energy_price_buy_per_kwh"] / 1000

    logger.info("Creating evaluation environment")
    eval_env = InverterEnvBatteryMgmt(
        inverter_sim=copy.deepcopy(i_sim),
        max_steps=full_period,
        reward_near_full=rewards["reward_near_full"],
        penalty_below_night_reserve=rewards["penalty_below_night_reserve"],
        penalty_below_min_reserve=rewards["penalty_below_min_reserve"],
        min_reserve=reserves["min_reserve"],
        night_reserver=reserves["night_reserver"],
        near_full=reserves["near_full"],
    )
    eval_env.shuffle = train_cfg["shuffle"]
    eval_env.inverter_sim.grid_sim.energy_price_buy = rewards["energy_price_buy_per_kwh"] / 1000

    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    num_envs = train_cfg["num_envs"]
    logger.info(f"Vectorizing environment ({num_envs} parallel envs)")
    t_env = DummyVecEnv([lambda: copy.deepcopy(train_env) for _ in range(num_envs)])

    logger.info("Constructing DQN model")
    model = DQN(
        train_cfg["policy"],
        t_env,
        verbose=0,
        learning_rate=train_cfg["learning_rate"],
        batch_size=train_cfg["batch_size"],
        device=train_cfg["device"],
    )

    logger.info("Setting up callbacks")
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(logs_dir, "best_model"),
        log_path=os.path.join(logs_dir, "results"),
        eval_freq=week * train_cfg["eval"]["eval_freq_multiplier"],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["checkpoint"]["save_freq"],
        save_path=os.path.join(logs_dir, "checkpoints"),
        name_prefix=train_cfg["checkpoint"]["name_prefix"],
        save_replay_buffer=train_cfg["checkpoint"]["save_replay_buffer"],
        save_vecnormalize=train_cfg["checkpoint"]["save_vecnormalize"],
    )

    total_timesteps = full_period * train_cfg["train_periods"]
    logger.info(f"Training for {total_timesteps} timesteps")
    logger.info("Starting model.learn()")

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])
    elapsed = time.time() - start
    logger.info(f"Training completed in {elapsed:.1f}s")

    model.save(os.path.join(logs_dir, "final_model"))
    logger.info(f"Model saved to {logs_dir}/final_model")
