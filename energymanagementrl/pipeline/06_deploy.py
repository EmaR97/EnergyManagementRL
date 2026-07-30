import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def run(config: dict):
    logger.info("Starting deployment")

    from .config import get_env, get_plant_config
    from ..utility import get_logger
    from ..production_forecast import EnergyPredictionSystem, OpenMeteoClient
    from ..fusion_solar_connector import FusionSolarClientParsed, PeriodicTask
    from ..rl import load_model_with_weights
    from ..rl.real_system_interaction import EnergyManagementSystem
    from ..interface import TelegramBot

    import gymnasium as gym
    from gymnasium import spaces

    plant_config = get_plant_config(config)
    production_forecaster = EnergyPredictionSystem(
        plant_config=plant_config, open_meteo_client=OpenMeteoClient()
    )

    username = get_env("FUSION_SOLAR_CLIENT_USERNAME", required=True)
    password = get_env("FUSION_SOLAR_CLIENT_PASSWORD", required=True)
    plant_cfg = config["solar_plant"]

    client = FusionSolarClientParsed(
        username, password, huawei_subdomain=plant_cfg["inverter"]["huawei_subdomain"]
    )
    periodic_task = PeriodicTask(client.keep_alive)
    periodic_task.start()
    plant_id = client.get_plant_ids()[0]
    battery_id = client.get_battery_ids(plant_id)[0]

    env = gym.Env()
    env.action_space = spaces.Discrete(2)
    env.observation_space = spaces.Box(low=0, high=1000, shape=(55,), dtype=np.float64)

    model_path = os.path.join(
        config["data_paths"].get("trained_models", "../data/trained_models"),
        "models",
        config["models"]["current_used"],
    )
    model = load_model_with_weights(env, model_path)

    logger_esm = get_logger("ESM")
    system = EnergyManagementSystem(
        client=client,
        plant_id=plant_id,
        battery_id=battery_id,
        production_forecaster=production_forecaster,
        model=model,
        logger=logger_esm,
    )

    token = get_env("TELEGRAM_BOT_TOKEN_TEST", required=True)
    admin_id = int(get_env("TELEGRAM_ID", required=True))
    logger_tb = get_logger("TB")

    bot = TelegramBot(system=system, token=token, allowed_users=[admin_id], logger=logger_tb)

    import asyncio
    import nest_asyncio

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(bot.set_bot_commands())
    import threading

    async def run_control_loop():
        await system.control_loop()

    threading.Thread(target=lambda: asyncio.run(run_control_loop())).start()
    loop.run_until_complete(bot.run())
