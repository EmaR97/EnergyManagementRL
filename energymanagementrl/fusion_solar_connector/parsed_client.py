from typing import List, Tuple

import numpy as np
import pandas as pd

from .extended_client import FusionSolarClientExtended, FusionSolarExceptionExtended


class FusionSolarClientParsed(FusionSolarClientExtended):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_battery_day_stats_parsed(
            self,
            battery_id: str,
            query_time: int = None,
            signals: List[FusionSolarClientExtended.BatterySignal] = None,
    ) -> pd.DataFrame:
        battery_stats_df = pd.concat(
            [
                pd.DataFrame(element['pmDataList'])
                .loc[:, ['startTime', 'counterValue']]
                .rename(columns={'counterValue': element['name'], 'startTime': 'timestamp'})
                .set_index('timestamp')
                for element in self.get_battery_day_stats(battery_id, query_time, signals).values()
            ],
            axis=1
        )
        battery_stats_df.index = pd.to_datetime(battery_stats_df.index, unit='s')
        return battery_stats_df

    PLANT_STATS_KEYS = ['xAxis', "productPower", 'usePower', 'selfUsePower', 'chargePower', 'dischargePower']

    def get_plant_stats_parsed(
            self,
            plant_id: str,
            query_time: int = None,
            time_zone: int = 0,
            time_zone_str: str = "UTC",
            keys_to_keep=None,
            time_zone_str_convert: str = "UTC",
    ) -> dict:

        if keys_to_keep is None:
            keys_to_keep = FusionSolarClientParsed.PLANT_STATS_KEYS
        plant_stats = self.get_plant_stats(plant_id, query_time, time_zone, time_zone_str)
        plant_stats_df = pd.DataFrame.from_dict(
            {key: plant_stats[key] for key in keys_to_keep})
        plant_stats_df['timestamp'] = pd.to_datetime(plant_stats_df['xAxis'])
        plant_stats_df = (
            plant_stats_df.set_index('timestamp')
            .tz_localize(time_zone_str_convert, ambiguous=True, nonexistent='shift_forward')
            .tz_convert('UTC')
            .tz_localize(None)
            .drop(columns=['xAxis'])
        )
        return plant_stats_df

    MILLISECONDS_IN_A_DAY = 24 * 60 * 60 * 1000  # Milliseconds in one day
    INVALID_FLOAT = np.float64(1.7976931348623157e+308)
    TIMEZONE_UTC = 'UTC'

    def fetch_statistics(self, battery_id, plant_id, timestamp_millis, time_zone_str_convert):
        """Fetch battery and plant statistics for a given timestamp."""
        battery_stats = self.get_battery_day_stats_parsed(
            battery_id,
            timestamp_millis,
            signals=[self.BatterySignal.SOC],
        )
        plant_stats = self.get_plant_stats_parsed(
            plant_id,
            timestamp_millis,
            time_zone=0,
            time_zone_str=FusionSolarClientParsed.TIMEZONE_UTC,
            time_zone_str_convert=time_zone_str_convert,
        )
        return pd.concat([battery_stats, plant_stats], axis=1)

    def get_plant_history(self, start_time, end_time, battery_id, plant_id, time_zone_str_convert) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Fetch and process historical plant data.
        Returns both unformatted and final cleaned plant history.
        """
        days = calculate_days(start_time, end_time)

        historical_data = [
            self.fetch_statistics(battery_id, plant_id, timestamp, time_zone_str_convert) for timestamp in days
        ]
        unformatted_data = pd.concat(historical_data, axis=0)
        cleaned_data = clean_and_format_data(unformatted_data)
        data_with_calculations = add_calculated_columns(cleaned_data)
        final_data = filter_final_data(data_with_calculations, end_time, time_zone_str_convert)

        return unformatted_data, final_data

    def get_plant_flow_parsed(self, plant_id: str) -> Tuple[float, float, float, float, float]:
        """
        Fetches and processes plant flow data from the client.

        Args:
            self: The client instance used to fetch plant data.
            plant_id (str): The ID of the plant.

        Returns:
            Tuple containing production, load, storage, and grid flow as floats.
        """
        # Fetch plant flow data
        flow_data = self.get_plant_flow(plant_id).get('data', {}).get('flow', {})

        # Extract specific elements from flow data
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
        if '--' in (string_prod, string_load, string_store, string_grid, string_soc):
            raise FusionSolarExceptionExtended(
                message=f"get_plant_flow_parsed",
                code=FusionSolarExceptionExtended.ErrorCode.PARSING
            )
        prod = float(string_prod)
        load = -float(string_load)
        store = float(string_store)
        grid = float(string_grid)
        soc = float(string_soc)
        # Adjust store and grid to maintain system balance
        store, grid = get_system_balance(prod, load, store, grid)

        return prod, load, store, grid, soc


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

    raise FusionSolarExceptionExtended(
        message=f"get_system_balance:{prod, load, store, grid}",
        code=FusionSolarExceptionExtended.ErrorCode.PARSING
    )
    # If no balance was found, return None or an appropriate value


def calculate_days(start_time, end_time):
    """Calculate list of timestamps in milliseconds for each day between start_time and end_time."""
    days_since_connection = (end_time - start_time).days
    start_of_today_millis = int(pd.to_datetime(end_time.strftime("%Y-%m-%d")).timestamp()) * 1000
    return [
        start_of_today_millis - (FusionSolarClientParsed.MILLISECONDS_IN_A_DAY * day_offset)
        for day_offset in reversed(range(days_since_connection))
    ]


def clean_and_format_data(unformatted_data):
    """Clean and format raw plant history data."""
    cleaned_data = (
        unformatted_data
        .replace(FusionSolarClientParsed.INVALID_FLOAT, np.nan)  # Replace invalid float values
        .replace('--', np.nan)  # Replace placeholder strings
        .astype(np.float16)  # Convert to float16 for memory efficiency
    )
    return cleaned_data


def add_calculated_columns(data):
    """Add calculated columns to plant history data."""
    data['stored_power_kw'] = data['dischargePower'] - data['chargePower']
    data['load_power_kw'] = -data['usePower']
    data['production_power_kw'] = data['productPower']
    data['grid_power_kw'] = (
            data['production_power_kw'] + data['stored_power_kw'] + data['load_power_kw']
    )
    return data


def filter_final_data(data, end_time, time_zone_str_convert):
    """Filter and select relevant columns for the final plant history."""
    data = data[
        ['production_power_kw', 'load_power_kw', 'grid_power_kw', 'stored_power_kw', 'SOC']
    ]
    end_time_utc = (
        end_time
        .tz_localize(time_zone_str_convert, ambiguous=True, nonexistent='shift_forward')
        .tz_convert(FusionSolarClientParsed.TIMEZONE_UTC)
        .tz_localize(None)
    )
    return data[data.index <= end_time_utc]
