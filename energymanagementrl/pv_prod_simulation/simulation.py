# simulation.py

from typing import List

import pandas as pd
from pvlib import modelchain, location
from pvlib import pvsystem


class PanelModel:

    def __init__(
            self,
            pdc0: float,
            temp_model_a: float,
            temp_model_b: float,
            delta_t: float,
            gamma_pdc: float
    ):
        self.pdc0 = pdc0
        self.temp_model_a = temp_model_a
        self.temp_model_b = temp_model_b
        self.delta_t = delta_t
        self.gamma_pdc = gamma_pdc


class ArrayConfig:

    def __init__(
            self,
            name: str,
            panel_model: PanelModel,
            num_panels: int,
            tilt_angle: float,
            azimuth: float
    ):
        self.name = name
        self.panel_model = panel_model
        self.num_panels = num_panels
        self.tilt_angle = tilt_angle
        self.azimuth = azimuth


class PlantConfig:

    def __init__(
            self,
            latitude: float,
            longitude: float,
            timezone: str,
            inverter_pdc0: float,
            arrays: List[ArrayConfig]
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.inverter_pdc0 = inverter_pdc0
        self.arrays = arrays

    def setup_pv_system(
            self
    ) -> pvsystem.PVSystem:
        arrays = []
        for array_config in self.arrays:
            array = pvsystem.Array(
                mount=pvsystem.FixedMount(array_config.tilt_angle, array_config.azimuth), module_parameters={
                    "pdc0": array_config.panel_model.pdc0 * array_config.num_panels,
                    "gamma_pdc": array_config.panel_model.gamma_pdc
                }, temperature_model_parameters={
                    "a": array_config.panel_model.temp_model_a,
                    "b": array_config.panel_model.temp_model_b,
                    "deltaT": array_config.panel_model.delta_t
                }, name=array_config.name
            )
            arrays.append(array)
        return pvsystem.PVSystem(
            arrays=arrays, inverter_parameters={
                "pdc0": self.inverter_pdc0
            }
        )


def run_simulation(
        mc,
        system,
        weather_data: pd.DataFrame
):
    mc.run_model(weather_data)
    dc_results = pd.concat(
        [mc.results.dc[i].rename(f'{array.name}_dc') for i, array in enumerate(system.arrays)], axis=1
    )
    ac_results = mc.results.ac.rename('inverter_ac')
    return pd.concat([dc_results, ac_results], axis=1)


def model_config(
        plant_config: PlantConfig
) -> tuple[modelchain.ModelChain, pvsystem.PVSystem, location.Location]:
    system = plant_config.setup_pv_system()
    loc = location.Location(plant_config.latitude, plant_config.longitude)
    mc = modelchain.ModelChain(system, loc, aoi_model='physical', spectral_model='no_loss')
    return mc, system, loc
