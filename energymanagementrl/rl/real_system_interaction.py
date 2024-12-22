import logging
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
        prod_kw, load_kw, charge_kw, grid_kw, soc = self.client.get_plant_flow_parsed(self.plant_id)
        prod_kwh, load_kwh, charge_kwh, grid_kwh = [i / 12 for i in (prod_kw, -load_kw, charge_kw, grid_kw)]
        stored_kwh = (soc - self.battery_min_percentage) / 100 * self.battery_capacity_kw
        return prod_kwh, load_kwh, charge_kwh, grid_kwh, stored_kwh

    def get_history(self):
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
            history.fillna(history.mean(), inplace=True)
        load_kwh_last_2day = history.resample('4h').mean()[1:] / 12
        return load_kwh_last_2day

    def compute_production_residual(self):
        now_gmt = pd.Timestamp.now(tz='UTC').to_pydatetime()
        now_gmt_5m = now_gmt.replace(minute=now_gmt.minute // 5 * 5)
        start = now_gmt_5m.strftime('%Y-%m-%d %H:%M')
        end = (now_gmt_5m + timedelta(days=2)).strftime('%Y-%m-%d %H:%M')
        clear_sky_df = self.production_forecaster.run_energy_production_prediction(start, end,
                                                                                   WeatherType.clear_sky) / 12
        open_meteo_df = self.production_forecaster.run_energy_production_prediction(start, end,
                                                                                    WeatherType.open_meteo_forecast) / 12
        forecast_range = [i * 12 for i in range(48)]
        residual = abs(clear_sky_df.inverter_ac - open_meteo_df.inverter_ac)
        prod_kwh_next_2day, residual_kwh_next_2day = (
            np.array(column.iloc[forecast_range]) @ sparse_matrix
            for column in (open_meteo_df.inverter_ac, residual)
        )
        return prod_kwh_next_2day, residual_kwh_next_2day

    def build_system_state(self, prod_kwh: float, load_kwh: float, charge_kwh: float, grid_kwh: float,
                           stored_kwh: float, load_kwh_last_2day,
                           prod_kwh_next_2day, residual_kwh_next_2day) -> dict[str, any]:
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

        prod_kwh, load_kwh, charge_kwh, grid_kwh, stored_kwh = self.get_flow_and_energy()
        load_kwh_last_2day = self.get_history()
        prod_kwh_next_2day, residual_kwh_next_2day = self.compute_production_residual()

        return self.build_system_state(
            prod_kwh, load_kwh, charge_kwh, grid_kwh, stored_kwh, load_kwh_last_2day,
            prod_kwh_next_2day, residual_kwh_next_2day
        )

    def execute_control(self, active: bool = False):
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
            current_state = (
                FusionSolarClientParsed.BatteryWorkingMode.MAXIMUM_SELF_CONSUMPTION
                if int(self.client.get_battery_status(self.battery_id)[1]['realValue']) == '4' else
                FusionSolarClientParsed.BatteryWorkingMode.FULLY_FEED_TO_GRID
            )
            if battery_mode.value != current_state:
                self.client.set_battery_working_mode(self.battery_id, battery_mode)
        logging.warning(f"Battery Mode: {battery_mode.name}")

        return state, action

    def control_loop(self, active: bool = False):
        try:
            while True:
                try:
                    state, action = self.execute_control(active=active)
                except FusionSolarExceptionExtended as e:
                    logging.error(f"Error: {e.code}")
                    sleep(10)
                    continue

                now = datetime.now()
                state.update({
                    'timestamp': now.timestamp(),
                    'action': int(action)
                })
                logging.info(f"State :{state}")

                next_time = (now + timedelta(minutes=5 - now.minute % 5)).replace(second=0, microsecond=0)
                sleep_duration = (next_time - now).total_seconds()

                sleep(sleep_duration)

        finally:
            if active:
                self.client.set_battery_working_mode(
                    self.battery_id,
                    FusionSolarClientParsed.BatteryWorkingMode.MAXIMUM_SELF_CONSUMPTION
                )
            logging.warning("Control loop terminated, battery mode reset.")
