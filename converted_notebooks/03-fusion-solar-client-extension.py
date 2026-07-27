# %%
# !pip install fusion_solar_py -q
# %%
# from kaggle_secrets import UserSecretsClient

# user_secrets = UserSecretsClient()
# FUSION_SOLAR_CLIENT_PASSWORD = user_secrets.get_secret("FUSION_SOLAR_CLIENT_PASSWORD")
# FUSION_SOLAR_CLIENT_USERNAME = user_secrets.get_secret("FUSION_SOLAR_CLIENT_USERNAME")
# %%
from dotenv import load_dotenv
import os

load_dotenv()
FUSION_SOLAR_CLIENT_PASSWORD = os.environ.get("FUSION_SOLAR_CLIENT_PASSWORD")
FUSION_SOLAR_CLIENT_USERNAME = os.environ.get("FUSION_SOLAR_CLIENT_USERNAME")
# %%
import pandas as pd
from energymanagementrl.fusion_solar_connector import FusionSolarClientParsed
# %%
# log into the API - with proper credentials...
client = FusionSolarClientParsed(FUSION_SOLAR_CLIENT_USERNAME, FUSION_SOLAR_CLIENT_PASSWORD,
                                 huawei_subdomain="uni004eu5")
_plant_id = client.get_plant_ids()[0]
_battery_id = client.get_battery_ids(_plant_id)[0]
# %%
client.set_battery_working_mode(_battery_id,client.BatteryWorkingMode.MAXIMUM_SELF_CONSUMPTION)
# %%
plant_data = client.get_plant_stats(_plant_id)
# %%
client.get_battery_day_stats(_battery_id)
# %%
current_time = pd.Timestamp.now()
grid_connection_time = pd.to_datetime(client.get_plant_details(_plant_id)['gridConnectedTime'].split()[0])
TIMEZONE_LOCAL = 'Europe/Rome'

unformatted_plant_history, final_plant_history = client.get_plant_history(grid_connection_time, current_time,
                                                                          _battery_id, _plant_id, TIMEZONE_LOCAL)
# %%
# Display the resulting dataframe
final_plant_history.info()
final_plant_history