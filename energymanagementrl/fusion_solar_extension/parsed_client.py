from typing import List, Tuple

import pandas as pd

from .extended_client import FusionSolarClientExtended


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
            .tz_localize(time_zone_str_convert, ambiguous=True)
            .tz_convert('UTC')
            .tz_localize(None)
            .drop(columns=['xAxis'])
        )
        return plant_stats_df

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
        prod = float(prod_data['description']['value'].split()[0])
        load = -float(load_data['description']['value'].split()[0])
        store = float(store_data['description']['value'].split()[0])
        grid = float(grid_data['description']['value'].split()[0])
        soc = float(store_data['deviceTips']['SOC'])
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

    # If no balance was found, return None or an appropriate value
    return None
