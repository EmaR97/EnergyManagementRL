# %%
from dotenv import load_dotenv
import os

load_dotenv()
LAT = float(os.environ.get("LAT"))
LON = float(os.environ.get("LON"))
# %%
import pandas as pd

from energymanagementrl.production_forecast import PanelModel, ArrayConfig, PlantConfig, WeatherType, \
    analyze_production, EnergyPredictionSystem, OpenMeteoClient
# %%
open_meteo_client = OpenMeteoClient()
# %%
panel_model = PanelModel(pdc0=0.42, temp_model_a=-3.56, temp_model_b=-0.075, delta_t=3, gamma_pdc=-0.004)
num_panels = 14
arrays = [ArrayConfig(name='sud_east', panel_model=panel_model, num_panels=num_panels, tilt_angle=25, azimuth=110),
          ArrayConfig(name='nord_west', panel_model=panel_model, num_panels=num_panels, tilt_angle=18, azimuth=290)]

plant_config = PlantConfig(
    latitude=LAT, longitude=LON, timezone='Europe/Rome', inverter_pdc0=6, arrays=arrays
)
energy_prediction_system = EnergyPredictionSystem(plant_config=plant_config, open_meteo_client=open_meteo_client)
# %%
s = '2024-12-24 00:05'
e = '2024-12-24 23:55'
open_meteo_production_df = energy_prediction_system.run_energy_production_prediction(
    s,
    e,
    WeatherType.open_meteo
)
open_meteo_production_df.info()
#
clear_sky_production_df = energy_prediction_system.run_energy_production_prediction(
    s,
    e,
    WeatherType.clear_sky
)
clear_sky_production_df.info()
analyze_production(clear_sky_production_df,1/12)
analyze_production(open_meteo_production_df,1/12)

# %%
clear_sky_production_df.rename({'inverter_ac': 'production_power_kw_optimal'}, axis=1, inplace=True)
open_meteo_production_df.rename({'inverter_ac': 'production_power_kw_weather_dependent'}, axis=1, inplace=True)
merged_df = pd.concat([clear_sky_production_df.production_power_kw_optimal,
                  open_meteo_production_df.production_power_kw_weather_dependent], axis=1)
merged_df.plot(figsize=(20, 6), grid=True)

# %%
