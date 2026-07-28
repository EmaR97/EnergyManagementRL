import json

import pytest

from energymanagementrl.pipeline.config import load_config, get_env, get_plant_config, get_simulation_params


class TestLoadConfig:
    def test_load_config(self, tmp_path, monkeypatch):
        config = {
            "solar_plant": {"timezone": "UTC"},
            "battery": {"capacity": 9000},
            "data_paths": {"base": "../data"},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))
        monkeypatch.setenv("ENERGY_MGMT_CONFIG", str(config_file))
        result = load_config()
        assert result["solar_plant"]["timezone"] == "UTC"

    def test_load_config_explicit_path(self, tmp_path):
        config = {"solar_plant": {"timezone": "UTC"}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))
        result = load_config(str(config_file))
        assert result["solar_plant"]["timezone"] == "UTC"

    def test_missing_config_raises(self):
        with pytest.raises(EnvironmentError):
            load_config("/nonexistent/config.json")


class TestGetEnv:
    def test_get_env_existing(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_123", "hello")
        assert get_env("TEST_VAR_123") == "hello"

    def test_get_env_missing_optional(self):
        assert get_env("NONEXISTENT_VAR_123") is None

    def test_get_env_missing_required(self):
        with pytest.raises(EnvironmentError, match="Required environment variable"):
            get_env("NONEXISTENT_VAR_123", required=True)


class TestGetPlantConfig:
    def test_builds_plant_config(self, sample_config, monkeypatch):
        monkeypatch.setenv("LAT", "45.0")
        monkeypatch.setenv("LON", "9.0")
        plant = get_plant_config(sample_config)
        assert plant.latitude == 45.0
        assert plant.longitude == 9.0
        assert len(plant.arrays) == 2
        assert plant.arrays[0].name == "sud_east"


class TestGetSimulationParams:
    def test_extracts_params(self, sample_config):
        params = get_simulation_params(sample_config)
        assert "battery" in params
        assert "grid" in params
        assert "simulation" in params
