# %%
# !pip -q install  fusion_solar_py pvlib retry-requests openmeteo_requests requests-cache python-telegram-bot
# %%
# !pip -q install git+https://github.com/EmaR97/EnergyManagementRL.git@testing-10
# %%
import os
import sys

from dotenv import load_dotenv


sys.path.append("..")
load_dotenv()


def get_env(name):
    return os.environ.get(name)
# %%
# from kaggle_secrets import UserSecretsClient
# 
# user_secrets = UserSecretsClient()
# 
# def get_env(name):
#     return user_secrets.get_secret(name)
# %%
FUSION_SOLAR_CLIENT_PASSWORD = get_env("FUSION_SOLAR_CLIENT_PASSWORD")
FUSION_SOLAR_CLIENT_USERNAME = get_env("FUSION_SOLAR_CLIENT_USERNAME")
LAT = float(get_env("LAT"))
LON = float(get_env("LON"))
TOKEN = get_env("TELEGRAM_BOT_TOKEN_TEST")
# TOKEN = get_env("TELEGRAM_BOT_TOKEN")
ADMIN_ID = int(get_env("TELEGRAM_ID"))
KAGGLE_USERNAME = get_env("KAGGLE_USERNAME")
# DATASET_NAME = "logs-persistence-test"
# DATASET_FILE = "logs.csv"
# METADATA_FILE = "dataset-metadata.json"
# DATASET_ID = f"{KAGGLE_USERNAME}/{DATASET_NAME}"
# DATASET_DIR = f"./{DATASET_NAME}"
WEBHOOK_URL = get_env("GOOGLE_APP_SCRIPT_LOGGER_URL")
KAGGLE = "kaggle"

# %%
from energymanagementrl.utility import get_logger

logger_main = get_logger("Main")
logger_esm = get_logger("ESM")
logger_tb = get_logger("TB")
# %%
from energymanagementrl.production_forecast import *

panel_model = PanelModel(pdc0=0.42, temp_model_a=-3.56, temp_model_b=-0.075, delta_t=3, gamma_pdc=-0.004)
num_panels = 14
arrays = [ArrayConfig(name='sud_east', panel_model=panel_model, num_panels=num_panels, tilt_angle=25, azimuth=110),
          ArrayConfig(name='nord_west', panel_model=panel_model, num_panels=num_panels, tilt_angle=18, azimuth=290)]

_plant_config = PlantConfig(
    latitude=LAT, longitude=LON, timezone='Europe/Rome', inverter_pdc0=6, arrays=arrays
)
_production_forecaster = EnergyPredictionSystem(plant_config=_plant_config, open_meteo_client=OpenMeteoClient())
# %%
from energymanagementrl.fusion_solar_connector import *

_client = FusionSolarClientParsed(FUSION_SOLAR_CLIENT_USERNAME, FUSION_SOLAR_CLIENT_PASSWORD,
                                  huawei_subdomain="uni004eu5")
periodic_task = PeriodicTask(_client.keep_alive)
periodic_task.start()
_plant_id = _client.get_plant_ids()[0]
battery_id = _client.get_battery_ids(_plant_id)[0]
# %%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from energymanagementrl.rl import load_model_with_weights

# Create a simple dummy environment used only to define the model struct, as the model will be used in inference
env = gym.Env()
env.action_space = spaces.Discrete(2)  # Two possible actions: 0 or 1
env.observation_space = spaces.Box(low=0, high=1000, shape=(55,), dtype=np.float64)  # 55 state variables

# Load the DQN policy
best_model = '../data/trained_models/models/dqn_1.0_0.06_0.06_0.02_1000_l_2.policy_weights.pth'
_model = load_model_with_weights(
    env,
    best_model,
    # 'dqn.2024_12_31_11_44_29.latest'
)
# %%
from energymanagementrl.rl.real_system_interaction import EnergyManagementSystem

system = EnergyManagementSystem(
    client=_client,
    plant_id=_plant_id,
    battery_id=battery_id,
    production_forecaster=_production_forecaster,
    model=_model,
    logger=logger_esm,
)

# system.set_active(True)
# %%
from energymanagementrl.interface import TelegramBot

bot = TelegramBot(
    system=system,
    token=TOKEN,
    allowed_users=[ADMIN_ID],
    logger=logger_tb,
)
# %%
import asyncio
import nest_asyncio

nest_asyncio.apply()
loop = asyncio.get_event_loop()
loop.run_until_complete(bot.set_bot_commands())
# %%
from energymanagementrl.utility.logger_to_google_app_script import LoggerToRemotePost
from energymanagementrl.utility.logger_to_kaggle_dataset import LoggerToKaggleDataset
import threading


async def run_control_loop():
    # with LoggerToKaggleDataset(logger_esm, DATASET_ID, DATASET_DIR, DATASET_FILE, METADATA_FILE) as log:
    with LoggerToRemotePost(logger_esm, WEBHOOK_URL, sheet_key=KAGGLE) as log:
        await system.control_loop()


threading.Thread(target=lambda: asyncio.run(run_control_loop())).start()
# %%
from datetime import datetime


async def shutdown(hour: int = None, minute: int = None):
    await sleep_async(hour, minute)
    await stopping_all()


async def stopping_all():
    logger_main.warning("Stopping all...")
    await system.stop_control_loop()
    bot.app.stop_running()


async def sleep_async(hour: int = None, minute: int = None):
    hour = hour or 11 if datetime.now().hour < 12 else 23
    minute = minute or 55
    now = datetime.now()
    stop_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    seconds = (stop_time - now).total_seconds()
    logger_main.info(f"Scheduled sleep until {stop_time}, sleeping for {seconds} seconds")
    await asyncio.sleep(seconds)
    logger_main.info(f"Scheduled sleep completed")


loop.create_task(
    # shutdown()
    shutdown(9,35)
)
# %%
loop.run_until_complete(bot.run())
# %%
