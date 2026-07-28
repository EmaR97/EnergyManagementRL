import pandas as pd
import pytest

from energymanagementrl.production_forecast import (
    PanelModel,
    ArrayConfig,
    PlantConfig,
    WeatherType,
    EnergyPredictionSystem,
)


def _make_plant_config():
    panel = PanelModel(pdc0=0.42, temp_model_a=-3.56, temp_model_b=-0.075, delta_t=3, gamma_pdc=-0.004)
    arrays = [
        ArrayConfig(name="sud_east", panel_model=panel, num_panels=7, tilt_angle=25, azimuth=110),
        ArrayConfig(name="nord_west", panel_model=panel, num_panels=7, tilt_angle=18, azimuth=290),
    ]
    return PlantConfig(latitude=45.0, longitude=9.0, timezone="Europe/Rome", inverter_pdc0=8, arrays=arrays)


class TestEnergyPredictionSystem:
    def test_clear_sky_simulation(self):
        plant = _make_plant_config()
        eps = EnergyPredictionSystem(plant_config=plant)
        result = eps.run_energy_production_prediction(
            "2024-06-15 08:00", "2024-06-15 18:00", WeatherType.clear_sky
        )
        assert isinstance(result, pd.DataFrame)
        assert "inverter_ac" in result.columns
        assert len(result) > 0

    def test_clear_sky_produces_energy(self):
        plant = _make_plant_config()
        eps = EnergyPredictionSystem(plant_config=plant)
        result = eps.run_energy_production_prediction(
            "2024-06-15 10:00", "2024-06-15 14:00", WeatherType.clear_sky
        )
        assert (result["inverter_ac"] > 0).any()

    def test_open_meteo_without_client_raises(self):
        plant = _make_plant_config()
        eps = EnergyPredictionSystem(plant_config=plant)
        with pytest.raises(ValueError, match="open meteo client"):
            eps.run_energy_production_prediction(
                "2024-06-15 08:00", "2024-06-15 18:00", WeatherType.open_meteo
            )

    def test_invalid_weather_type_raises(self):
        plant = _make_plant_config()
        eps = EnergyPredictionSystem(plant_config=plant)
        with pytest.raises(ValueError, match="Invalid weather type"):
            eps.run_energy_production_prediction(
                "2024-06-15 08:00", "2024-06-15 18:00", "invalid"
            )

    def test_run_simulation_directly(self):
        plant = _make_plant_config()
        eps = EnergyPredictionSystem(plant_config=plant)
        times = pd.date_range("2024-06-15 10:00", periods=12, freq="5min")
        location = eps.loc
        weather = location.get_clearsky(times)
        result = eps.run_simulation(weather)
        assert "inverter_ac" in result.columns
        assert len(result) == 12
