from typing import List, Tuple

import numpy as np
import pandas as pd
import pytz.exceptions
from pandas import DataFrame

from . import BatteryWorkingMode
from .extended_client import FusionSolarClientExtended, FusionSolarExceptionExtended


class FusionSolarClientParsed(FusionSolarClientExtended):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_day_start_sec(self) -> int:
        return self._get_day_start_sec()

    def reset_session(self):
        self._configure_session()

    def get_battery_day_stats_parsed(self, battery_id: str, query_time: int = None,
                                     signals: List[FusionSolarClientExtended.BatterySignal] | List[
                                         FusionSolarClientExtended.InverterSignal] = None, ) -> pd.DataFrame:
        battery_stats_df = pd.concat([pd.DataFrame(element['pmDataList']).loc[:, ['startTime', 'counterValue']].rename(
            columns={'counterValue': element['name'], 'startTime': 'timestamp'}).set_index('timestamp') for element in
                                      self.get_battery_day_stats(battery_id, query_time, signals).values()], axis=1)
        battery_stats_df.index = pd.to_datetime(battery_stats_df.index, unit='s')
        return battery_stats_df

    PLANT_STATS_KEYS = ['xAxis', "productPower", 'usePower', 'selfUsePower', 'chargePower', 'dischargePower']

    def get_plant_stats_parsed(self, plant_id: str, query_time: int = None, time_zone: int = 0,
                               time_zone_str: str = "UTC", keys_to_keep=None,
                               time_zone_str_convert: str = "UTC", ) -> DataFrame:

        if keys_to_keep is None:
            keys_to_keep = FusionSolarClientParsed.PLANT_STATS_KEYS
        plant_stats = self.get_plant_stats(plant_id, query_time, time_zone, time_zone_str)
        plant_stats_df = pd.DataFrame.from_dict({key: plant_stats[key] for key in keys_to_keep})
        plant_stats_df['timestamp'] = pd.to_datetime(plant_stats_df['xAxis'])
        try:
            plant_stats_df = (plant_stats_df.set_index('timestamp').tz_localize(time_zone_str_convert, ambiguous="NaT",
                                                                                nonexistent='shift_forward').tz_convert(
                'UTC').tz_localize(None).drop(columns=['xAxis']))
        except pytz.exceptions.AmbiguousTimeError as e:
            print("AmbiguousTimeError:", e)
            ambiguous_times = plant_stats_df[plant_stats_df['timestamp'].dt.hour == 2]
            print("Ambiguous times detected:", ambiguous_times)
            # Optionally drop ambiguous rows or handle them as needed
            plant_stats_df = plant_stats_df[plant_stats_df['timestamp'].dt.hour != 2]
        finally:
            pass
        return plant_stats_df

    MILLISECONDS_IN_A_DAY = 24 * 60 * 60 * 1000  # Milliseconds in one day
    INVALID_FLOAT = np.float64(1.7976931348623157e+308)
    TIMEZONE_UTC = 'UTC'

    def fetch_statistics(self, battery_id, inverter_id, plant_id, timestamp_millis, time_zone_str_convert):
        """Fetch battery and plant statistics for a given timestamp."""
        battery_stats = self.get_battery_day_stats_parsed(battery_id, timestamp_millis,
                                                          signals=[self.BatterySignal.SOC], )
        inverter_stats = self.get_battery_day_stats_parsed(inverter_id, timestamp_millis,
                                                           signals=[self.InverterSignal.GRID_VOLTAGE], )
        plant_stats = self.get_plant_stats_parsed(plant_id, timestamp_millis,  # time_zone=0,
                                                  time_zone_str=FusionSolarClientParsed.TIMEZONE_UTC,
                                                  time_zone_str_convert=time_zone_str_convert, )
        battery_stats = battery_stats.reset_index()
        inverter_stats = inverter_stats.reset_index()
        plant_stats = plant_stats.reset_index()
        battery_stats = battery_stats.drop_duplicates(subset='timestamp')
        inverter_stats = inverter_stats.drop_duplicates(subset='timestamp')
        plant_stats = plant_stats.drop_duplicates(subset='timestamp')
        inverter_stats = inverter_stats.set_index("timestamp")
        battery_stats = battery_stats.set_index("timestamp")
        plant_stats = plant_stats.set_index("timestamp")
        return pd.concat([battery_stats, inverter_stats, plant_stats], axis=1)

    def get_plant_history(self, start_time, end_time, battery_id, inverter_id, plant_id, time_zone_str_convert) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Fetch and process historical plant data.
        Iterates from newest to oldest day. Stops early when it finds a full week
        where all columns except SOC are '--' (no data available).
        Returns both unformatted and final cleaned plant history.
        """
        days = calculate_days(start_time, end_time)
        EMPTY_WEEK_THRESHOLD = 7

        empty_streak = 0
        historical_data = []

        for timestamp in reversed(days):
            day_df = self.fetch_statistics(battery_id, inverter_id, plant_id, timestamp, time_zone_str_convert)

            non_soc_cols = [c for c in day_df.columns if c != 'SOC']
            is_empty_day = all((day_df[c] == '--').all() for c in non_soc_cols)

            if is_empty_day:
                empty_streak += 1
                if empty_streak >= EMPTY_WEEK_THRESHOLD:
                    break
            else:
                empty_streak = 0
                historical_data.append(day_df)

        historical_data.reverse()

        if not historical_data:
            empty = pd.DataFrame()
            return empty, empty

        unformatted_data = pd.concat(historical_data, axis=0)
        cleaned_data = clean_and_format_data(unformatted_data)
        data_with_calculations = add_calculated_columns(cleaned_data)
        final_data = filter_final_data(data_with_calculations, end_time, time_zone_str_convert)

        return unformatted_data, final_data

    def get_plant_flow_parsed(self, plant_id: str) -> dict:
        flow_data = self.get_plant_flow(plant_id).get('data', {}).get('flow', {})

        try:
            grid_data = flow_data['links'][5]
            prod_data = flow_data['nodes'][0]
            store_data = flow_data['nodes'][4]
            load_data = flow_data['nodes'][5]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected structure in plant data: {e}")
        # Parse numeric values from descriptions
        string_prod = prod_data['description']['value'].split()[0]
        string_load = load_data['description']['value'].split()[0]
        string_store = store_data['description']['value'].split()[0]
        string_grid = grid_data['description']['value'].split()[0]
        string_soc = store_data['deviceTips']['SOC']
        string_cmv = store_data['deviceTips']['CHARGE_MODE_VALUE']
        if '--' in (string_prod, string_load, string_store, string_grid, string_soc, string_cmv):
            raise FusionSolarExceptionExtended(message=f"get_plant_flow_parsed",
                                               code=FusionSolarExceptionExtended.ErrorCode.PARSING)
        prod = float(string_prod)
        load = -float(string_load)
        store = float(string_store)
        grid = float(string_grid)
        soc = float(string_soc)
        # Adjust store and grid to maintain system balance
        store, grid = get_system_balance(prod, load, store, grid)
        match string_cmv:
            case "Maximum self-consumption":
                battery_mode = BatteryWorkingMode.MAXIMUM_SELF_CONSUMPTION
            case "Fully fed to grid":
                battery_mode = BatteryWorkingMode.FULLY_FEED_TO_GRID
            case _:
                raise ValueError(f"Unexpected battery mode: {string_cmv}")

        return {"prod": prod, "load": load, "store": store, "grid": grid, "soc": soc, "battery_mode": battery_mode}


def get_system_balance(prod, load, store, grid, tolerance=1e-6):
    # Calculate the imbalance between production and consumption
    imbalance = prod + load

    # Check each possible combination of signs for store and grid
    for sign_C, sign_D in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        store_balance = store * sign_C
        grid_balance = grid * sign_D

        # If the balance matches the imbalance within a tolerance, return the results
        if abs(store_balance + grid_balance + imbalance) < tolerance:
            return store_balance, grid_balance

    raise FusionSolarExceptionExtended(message=f"get_system_balance:{prod, load, store, grid}",
                                       code=FusionSolarExceptionExtended.ErrorCode.PARSING)  # If no balance was found, return None or an appropriate value


def calculate_days(start_time, end_time):
    """Calculate list of timestamps in milliseconds for each day between start_time and end_time."""
    days_since_connection = (end_time - start_time).days
    start_of_today_millis = int(pd.to_datetime(end_time.strftime("%Y-%m-%d")).timestamp()) * 1000
    return [start_of_today_millis - (FusionSolarClientParsed.MILLISECONDS_IN_A_DAY * day_offset) for day_offset in
            reversed(range(days_since_connection))]


def clean_and_format_data(unformatted_data):
    """Clean and format raw plant history data."""
    cleaned_data = (
        unformatted_data.replace(FusionSolarClientParsed.INVALID_FLOAT, np.nan)  # Replace invalid float values
        .replace('--', np.nan)  # Replace placeholder strings
        .astype(np.float16)  # Convert to float16 for memory efficiency
    )
    return cleaned_data


def add_calculated_columns(data):
    """Add calculated columns to plant history data."""
    data['stored_power_kw'] = data['dischargePower'] - data['chargePower']
    data['load_power_kw'] = -data['usePower']
    data['production_power_kw'] = data['productPower']
    data['grid_power_kw'] = (data['production_power_kw'] + data['stored_power_kw'] + data['load_power_kw'])
    return data


def filter_final_data(data, end_time, time_zone_str_convert):
    """Filter and select relevant columns for the final plant history."""
    data = data[['production_power_kw', 'load_power_kw', 'grid_power_kw', 'stored_power_kw', 'SOC','GRID_VOLTAGE']]
    end_time_utc = (end_time.tz_localize(time_zone_str_convert, ambiguous=True, nonexistent='shift_forward').tz_convert(
        FusionSolarClientParsed.TIMEZONE_UTC).tz_localize(None))
    return data[data.index <= end_time_utc]
