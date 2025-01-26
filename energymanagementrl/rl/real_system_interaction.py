import logging
from http.client import RemoteDisconnected

from requests.exceptions import ConnectionError

from time import sleep
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from energymanagementrl.fusion_solar_connector import *
from energymanagementrl.production_forecast import *
from energymanagementrl.simulation import sparse_matrix
from energymanagementrl.rl import extract_values_gen


class EnergyManagementSystem:
    """
    Manages energy flows and controls battery systems based on real-time data and predictions.

    Attributes:
        client: FusionSolarClientParsed instance for solar data access.
        plant_id: Unique identifier for the solar plant.
        battery_id: Unique identifier for the battery.
        production_forecaster: Instance for energy production forecasting.
        model: Reinforcement learning model for decision-making.
        battery_capacity_kw: Maximum battery capacity in kW.
        battery_min_percentage: Minimum state of charge as a percentage.
    """

    def __init__(
            self,
            client: FusionSolarClientParsed,
            plant_id: str,
            battery_id: str,
            production_forecaster: EnergyPredictionSystem,
            model: DQN,
            battery_capacity_kw: int = 10,
            battery_min_percentage: int = 10
    ):
        self.client: FusionSolarClientParsed = client
        self.plant_id: str = plant_id
        self.battery_id: str = battery_id
        self.production_forecaster: EnergyPredictionSystem = production_forecaster
        self.model: DQN = model
        self.battery_capacity_kw = battery_capacity_kw
        self.battery_min_percentage = battery_min_percentage

    def get_flow_and_energy(self):
        """Retrieve and calculate energy flow data from the plant."""
        prod_kw, load_kw, charge_kw, grid_kw, soc = self.client.get_plant_flow_parsed(self.plant_id)
        prod_kwh, load_kwh, charge_kwh, grid_kwh = [i / 12 for i in (prod_kw, -load_kw, charge_kw, grid_kw)]
        stored_kwh = (soc - self.battery_min_percentage) / 100 * self.battery_capacity_kw
        return prod_kwh, load_kwh, charge_kwh, grid_kwh, stored_kwh

    def get_history(self):
        """Retrieve and process historical load data."""
        now = datetime.now()
        history = pd.concat([
            self.client.get_plant_stats_parsed(
                self.plant_id,
                query_time=self.client._get_day_start_sec() + i * self.client.MILLISECONDS_IN_A_DAY,
                time_zone=2,
                time_zone_str='Europe/Rome'
            )
            for i in [-2, -1, 0]
        ]).usePower

        history = history[history.index <= now][-288 * 2:].replace('--', np.nan)
        history = history.astype('float')

        if history.isnull().values.any():
            logging.warning("Missing values in history data. Filling with mean.")
            history.fillna(history.mean(), inplace=True)

        load_kwh_last_2day = history.resample('4h').mean()[1:] / 12
        return load_kwh_last_2day

    def compute_production_residual(self):
        """Compute production residuals based on forecasts."""
        now_gmt = pd.Timestamp.now(tz='UTC').to_pydatetime()
        now_gmt_5m = now_gmt.replace(minute=now_gmt.minute // 5 * 5)
        start = now_gmt_5m.strftime('%Y-%m-%d %H:%M')
        end = (now_gmt_5m + timedelta(days=2)).strftime('%Y-%m-%d %H:%M')

        def get_forecast(weather_type):
            """Fetch energy production predictions for a given weather type."""
            return self.production_forecaster.run_energy_production_prediction(start, end, weather_type) / 12

        clear_sky_df = get_forecast(WeatherType.clear_sky)
        open_meteo_df = get_forecast(WeatherType.open_meteo_forecast)
        residual = abs(clear_sky_df.inverter_ac - open_meteo_df.inverter_ac)

        forecast_range = [i * 12 for i in range(48)]
        prod_kwh_next_2day = np.array(open_meteo_df.inverter_ac.iloc[forecast_range]) @ sparse_matrix
        residual_kwh_next_2day = np.array(residual.iloc[forecast_range]) @ sparse_matrix

        return prod_kwh_next_2day, residual_kwh_next_2day

    @staticmethod
    def _build_system_state(prod_kwh: float, load_kwh: float, charge_kwh: float, grid_kwh: float,
                            stored_kwh: float, load_kwh_last_2day,
                            prod_kwh_next_2day, residual_kwh_next_2day) -> dict[str, any]:
        """Construct the system state from energy data."""
        state = {
            'prod_sim': {},
            'cons_sim': {},
            'batt_sim': {},
            'grid_sim': {},
        }

        for i, value in enumerate(prod_kwh_next_2day):
            state['prod_sim'][f"energy_sample_{i}"] = round(float(value), 3)
        state["prod_sim"]["energy"] = prod_kwh

        for i, value in enumerate(residual_kwh_next_2day):
            state['prod_sim'][f"residual_sample_{i}"] = round(float(value), 3)

        for i, value in enumerate(load_kwh_last_2day):
            state['cons_sim'][f"energy_sample_{i}"] = round(float(value), 3)
        state["cons_sim"]["energy"] = round(load_kwh, 3)

        state["batt_sim"]["stored"] = round(stored_kwh, 3)
        state["batt_sim"]["charge_rate"] = -round(charge_kwh, 3) if charge_kwh < 0 else 0
        state["batt_sim"]["discharge_rate"] = round(charge_kwh, 3) if charge_kwh > 0 else 0

        state["grid_sim"]["feed_to_grid"] = -round(grid_kwh, 3) if grid_kwh < 0 else 0
        state["grid_sim"]["taken_from_grid"] = round(grid_kwh, 3) if grid_kwh > 0 else 0

        return state

    def get_system_state(self):
        """Retrieve and build the current system state."""
        prod_kwh, load_kwh, charge_kwh, grid_kwh, stored_kwh = self.get_flow_and_energy()
        load_kwh_last_2day = self.get_history()
        prod_kwh_next_2day, residual_kwh_next_2day = self.compute_production_residual()

        return self._build_system_state(
            prod_kwh, load_kwh, charge_kwh, grid_kwh, stored_kwh, load_kwh_last_2day,
            prod_kwh_next_2day, residual_kwh_next_2day
        )

    def execute_control(self, active: bool = False):
        """Execute a control decision using the RL model."""
        state = self.get_system_state()
        obs = np.array(list(extract_values_gen(state)))

        if len(obs) != 55:
            raise FusionSolarExceptionExtended(
                "Invalid observation length",
                FusionSolarExceptionExtended.ErrorCode.PARSING
            )

        action, _ = self.model.predict(obs)

        battery_mode = (
            FusionSolarClientParsed.BatteryWorkingMode.MAXIMUM_SELF_CONSUMPTION
            if action == 1 else
            FusionSolarClientParsed.BatteryWorkingMode.FULLY_FEED_TO_GRID
        )

        if active:
            try:
                current_mode = int(self.client.get_battery_status(self.battery_id)[1]['realValue'])
            except ValueError as e:
                logging.error(e)
                raise FusionSolarExceptionExtended('', FusionSolarExceptionExtended.ErrorCode.PARSING)

            current_state = (
                FusionSolarClientParsed.BatteryWorkingMode.MAXIMUM_SELF_CONSUMPTION
                if current_mode == '4' else
                FusionSolarClientParsed.BatteryWorkingMode.FULLY_FEED_TO_GRID
            )
            if battery_mode.value != current_state:
                self.client.set_battery_working_mode(self.battery_id, battery_mode)
        logging.warning(f"Battery Mode: {battery_mode.name}")

        return state, action

    def control_loop(self, active: bool = False, retry_delay: int = 10, retry_attempts: int = 10):
        """Run the control loop at 5-minute intervals."""
        try:
            while True:
                if self._is_scheduled_stop():
                    logging.info("Control loop stopping at scheduled time.")
                    break

                self._run_control_cycle(active, retry_delay)

        finally:
            logging.warning("Control loop terminated")
            if active:
                self._reset_battery_mode(retry_delay, retry_attempts)

    def _is_scheduled_stop(self) -> bool:
        """Check if the current time falls within the scheduled stop period."""
        now = datetime.now()
        return now.hour in (11, 23) and now.minute >= 55

    def _run_control_cycle(self, active: bool, retry_delay: int):
        """Execute a single control cycle."""
        try:
            state, action = self.execute_control(active=active)
        except (FusionSolarExceptionExtended, RemoteDisconnected, ConnectionError) as e:
            logging.error(f"Error: {getattr(e, 'code', str(e))}")
            sleep(retry_delay)
            return

        now = datetime.now()
        state.update({
            'timestamp': now.timestamp(),
            'action': int(action)
        })
        logging.info(f"State: {state}")

        next_time = (now + timedelta(minutes=5 - now.minute % 5)).replace(second=30, microsecond=0)
        sleep_duration = (next_time - now).total_seconds()
        sleep(sleep_duration)

    def _reset_battery_mode(self, retry_delay: int, retry_attempts: int):
        """Reset the battery mode with retries."""
        for attempt in range(retry_attempts):
            try:
                self.client.set_battery_working_mode(
                    self.battery_id,
                    FusionSolarClientParsed.BatteryWorkingMode.MAXIMUM_SELF_CONSUMPTION
                )
                logging.warning("Battery mode reset")
                break
            except (FusionSolarExceptionExtended, RemoteDisconnected, ConnectionError) as e:
                logging.error(f"Attempt-{attempt} failed. Error: {getattr(e, 'code', str(e))}")
                sleep(retry_delay)
            except Exception as e:
                logging.critical(f"Unexpected error during battery mode reset: {str(e)}")
                break
