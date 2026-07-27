# %%
from dotenv import load_dotenv
import os

load_dotenv()
LAT = float(os.environ.get("LAT"))
LON = float(os.environ.get("LON"))
FUSION_SOLAR_CLIENT_PASSWORD = os.environ.get("FUSION_SOLAR_CLIENT_PASSWORD")
FUSION_SOLAR_CLIENT_USERNAME = os.environ.get("FUSION_SOLAR_CLIENT_USERNAME")
input_dir = '../data/simulation-inputs/'

# Check if the directory exists
if not os.path.exists(input_dir):
    # Create the directory
    os.makedirs(input_dir)
    print(f"Directory '{input_dir}' created.")
else:
    print(f"Directory '{input_dir}' already exists.")
# %%
import pandas as pd
import sys
sys.path.append("..")
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
    latitude=LAT, longitude=LON, timezone='Europe/Rome', inverter_pdc0=7.5, arrays=arrays
)
energy_prediction_system = EnergyPredictionSystem(plant_config=plant_config, open_meteo_client=open_meteo_client)
# %%
client = FusionSolarClientParsed(FUSION_SOLAR_CLIENT_USERNAME, FUSION_SOLAR_CLIENT_PASSWORD,
                                 huawei_subdomain="uni004eu5")
_plant_id = client.get_plant_ids()[0]
_battery_id = client.get_battery_ids(_plant_id)[0]
_inverter_id='NE=150331722'
# %%
start = pd.Timestamp('2024-10-20 00:05')
end = pd.Timestamp.now().replace(hour=0,minute=0, second=0, microsecond=0)- pd.tseries.offsets.Minute()
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
                                                                          _battery_id,_inverter_id, _plant_id, TIMEZONE_LOCAL)
# %%
final_plant_history
# %%
partial_df = pd.concat([final_plant_history, merged_df], axis=1)
i = 0
partial_df[['production_power_kw_weather_dependent', 'production_power_kw_optimal', 'production_power_kw']][
2000 * i:2000 * (i + 1)].plot(figsize=(20, 6), grid=True)
# %%
# df=pd.read_csv(input_dir+'complete_series.ex.12kw.csv')['grid_voltage']
# %%
complete_df=pd.concat([partial_df.reset_index()],axis=1).set_index('index').dropna()
# %%
complete_df.to_csv(f'{input_dir}complete_series.{num_panels}panels.csv', index_label='index')
# %%
complete_df

# %%
!kaggle datasets version -p ../data/simulation-inputs/ -m "Added new dataset version"  --dir-mode zip