import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from ..production_forecast import PanelModel, ArrayConfig, PlantConfig


ENERGY_MGMT_CONFIG_ENV = "ENERGY_MGMT_CONFIG"

START_DATE = "2024-10-19"

def _configure_logging(log_level: str):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = os.environ.get(ENERGY_MGMT_CONFIG_ENV)
    if config_path is None:
        raise EnvironmentError(
            f"Config path not set. Either pass config_path or set {ENERGY_MGMT_CONFIG_ENV} env var"
        )

    config_path = Path(config_path)

    env_path = config_path.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    with open(config_path) as f:
        config = json.load(f)

    _resolve_data_paths(config, config_path.parent)
    _configure_logging(config.get("log_level", "INFO"))
    return config


def _resolve_data_paths(config: dict, config_dir: Path):
    data_paths = config.get("data_paths")
    if data_paths is None:
        return
    base = data_paths.get("base", "../data")
    if not os.path.isabs(base):
        data_paths["base"] = str((config_dir / base).resolve())
    for key, value in data_paths.items():
        if key == "base":
            continue
        if isinstance(value, str) and not os.path.isabs(value):
            data_paths[key] = str((config_dir / value).resolve())


def get_env(name: str, required: bool = False) -> str:
    value = os.environ.get(name)
    if required and value is None:
        raise EnvironmentError(f"Required environment variable not set: {name}")
    return value


def get_plant_config(config: dict) -> PlantConfig:
    plant_cfg = config["solar_plant"]
    panel_model = PanelModel(**plant_cfg["panel_model"])
    num_panels = plant_cfg["num_panels"]
    arrays = [
        ArrayConfig(
            name=a["name"],
            panel_model=panel_model,
            num_panels=num_panels,
            tilt_angle=a["tilt_angle"],
            azimuth=a["azimuth"],
        )
        for a in plant_cfg["arrays"]
    ]
    return PlantConfig(
        latitude=float(get_env("LAT", required=True)),
        longitude=float(get_env("LON", required=True)),
        timezone=plant_cfg["timezone"],
        inverter_pdc0=plant_cfg["inverter"]["pdc0"],
        arrays=arrays,
    )


def get_simulation_params(config: dict) -> dict:
    return {
        "battery": config["battery"],
        "grid": config["grid"],
        "simulation": config["simulation"],
    }

