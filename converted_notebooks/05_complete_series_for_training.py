# %%
# !pip -q install git+https://github.com/EmaR97/EnergyManagementRL.git@testing-7
# !pip -q install  fusion_solar_py pvlib retry-requests openmeteo_requests requests-cache
# !mkdir logs
# %%
# import os
# from kaggle_secrets import UserSecretsClient
# 
# # Retrieve the Kaggle credentials securely
# user_secrets = UserSecretsClient()
# os.environ['KAGGLE_USERNAME'] = user_secrets.get_secret("KAGGLE_USERNAME")
# os.environ['KAGGLE_KEY'] = user_secrets.get_secret("KAGGLE_KEY")
# %%
from dotenv import load_dotenv
import os

load_dotenv()
LAT = float(os.environ.get("LAT"))
LON = float(os.environ.get("LON"))
FUSION_SOLAR_CLIENT_PASSWORD = os.environ.get("FUSION_SOLAR_CLIENT_PASSWORD")
FUSION_SOLAR_CLIENT_USERNAME = os.environ.get("FUSION_SOLAR_CLIENT_USERNAME")
# %%
import pandas as pd

from energymanagementrl.production_forecast import PanelModel, ArrayConfig, PlantConfig, WeatherType, \
    EnergyPredictionSystem, OpenMeteoClient, analyze_production
from energymanagementrl.fusion_solar_connector import FusionSolarClientParsed
# %%
open_meteo_client = OpenMeteoClient()
panel_model = PanelModel(pdc0=0.42, temp_model_a=-3.56, temp_model_b=-0.075, delta_t=3, gamma_pdc=-0.004)
num_panels = 14
arrays = [ArrayConfig(name='sud_east', panel_model=panel_model, num_panels=num_panels, tilt_angle=25, azimuth=110),
          ArrayConfig(name='nord_west', panel_model=panel_model, num_panels=num_panels, tilt_angle=18, azimuth=290)]

plant_config = PlantConfig(
    latitude=LAT, longitude=LON, timezone='Europe/Rome', inverter_pdc0=6, arrays=arrays
)
energy_prediction_system = EnergyPredictionSystem(plant_config=plant_config, open_meteo_client=open_meteo_client)
# %%
client = FusionSolarClientParsed(FUSION_SOLAR_CLIENT_USERNAME, FUSION_SOLAR_CLIENT_PASSWORD,
                                 huawei_subdomain="uni004eu5")
_plant_id = client.get_plant_ids()[0]
_battery_id = client.get_battery_ids(_plant_id)[0]
# %%
s_start = '2024-10-20 00:05'
s_end = pd.Timestamp.now().replace(hour=0,minute=5, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M')

start = pd.Timestamp(
    s_start
    , tz='Europe/Rome'
).tz_convert('UTC').tz_convert(None)
end = pd.Timestamp(
    s_end
    , tz='Europe/Rome').tz_convert('UTC').tz_convert(None)
# %%
open_meteo_production_df = energy_prediction_system.run_energy_production_prediction(
    start.strftime('%Y-%m-%d %H:%M'),
    end.strftime('%Y-%m-%d %H:%M'),
    WeatherType.open_meteo
)
analyze_production(open_meteo_production_df,1/12)

clear_sky_production_df = energy_prediction_system.run_energy_production_prediction(
    start.strftime('%Y-%m-%d %H:%M'),
    end.strftime('%Y-%m-%d %H:%M'),
    WeatherType.clear_sky
)
analyze_production(clear_sky_production_df,1/12)
open_meteo_production_df.info()
clear_sky_production_df.info()
# %%
clear_sky_production_df.rename({'inverter_ac': 'production_power_kw_optimal'}, axis=1, inplace=True)
open_meteo_production_df.rename({'inverter_ac': 'production_power_kw_weather_dependent'}, axis=1, inplace=True)
merged_df = pd.concat([clear_sky_production_df.production_power_kw_optimal,
                       open_meteo_production_df.production_power_kw_weather_dependent], axis=1)
# %%
TIMEZONE_LOCAL = 'Europe/Rome'
unformatted_plant_history, final_plant_history = client.get_plant_history(pd.Timestamp(start), pd.Timestamp(end),
                                                                          _battery_id, _plant_id, TIMEZONE_LOCAL)
# %%
partial_df = pd.concat([final_plant_history, merged_df], axis=1)
i = 0
partial_df[['production_power_kw_weather_dependent', 'production_power_kw_optimal', 'production_power_kw']][
2000 * i:2000 * (i + 1)].plot(figsize=(20, 6), grid=True)
# %%
df=pd.read_csv('../data/model_inputs/complete_series.ex.12kw.csv')['grid_voltage']
# %%
complete_df=pd.concat([df,partial_df.reset_index()],axis=1).set_index('index').dropna()
# %%
complete_df.to_csv(f'../data/model_inputs/complete_series.{num_panels}panels.csv', index_label='index')
# %%
!kaggle datasets version -p ../data/model_inputs/ -m "Added new dataset version"  --dir-mode zip
# %%
