import time
from enum import Enum
from typing import List, Tuple

from fusion_solar_py.client import FusionSolarClient, logged_in
from fusion_solar_py.exceptions import FusionSolarException


class FusionSolarClientExtended(FusionSolarClient):
    """Extension of FusionSolarClient with additional functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    class BatteryWorkingMode(Enum):
        MAXIMUM_SELF_CONSUMPTION = 2
        FULLY_FEED_TO_GRID = 4

    @logged_in
    def set_battery_working_mode(self, battery_id, mode: BatteryWorkingMode):
        if not isinstance(mode, self.BatteryWorkingMode):
            raise ValueError(
                f"Invalid mode: {mode}. Expected one of {[e.name for e in self.BatteryWorkingMode]}"
            )

        url = f"https://{self._huawei_subdomain}.fusionsolar.huawei.com/rest/pvms/web/device/v1/deviceExt/set-config-signals"
        data = {
            "dn": battery_id,
            "changeValues": f'[{{"id":"230320241","value":"{mode.value}"}}]',
        }

        response = self._session.post(url, data=data)
        response.raise_for_status()

        try:
            response_json = response.json()
            for data in response_json['data']:
                if data['code'] != 0:
                    raise FusionSolarException(
                        f"Failed to set mode for battery {data['dn']}"
                    )
        except ValueError:
            print("Error: The response is not in JSON format.")

    @logged_in
    def get_plant_stats(
            self,
            plant_id: str,
            query_time: int = None,
            time_zone: int = 2,
            time_zone_str: str = "Europe/Vienna"
    ) -> dict:
        """
        Retrieves energy usage statistics for a specified plant.

        This method fetches the complete energy statistics for a given plant ID, either
        for the current day (default) or a specific date. The timestamps are expected
        to be in milliseconds.

        :param plant_id: The unique identifier of the plant.
        :type plant_id: str
        :param query_time: A timestamp representing the start of the day (00:00:00)
                           for which the data should be retrieved, in milliseconds.
                           Defaults to the current day if not provided.
        :type query_time: int, optional
        :param time_zone: The numerical representation of the time zone offset
                          from UTC (e.g., 2 for UTC+2). Defaults to 2.
        :type time_zone: int, optional
        :param time_zone_str: A string representing the time zone (e.g., "Europe/Vienna").
                              Defaults to "Europe/Vienna".
        :type time_zone_str: str, optional
        :return: A dictionary containing the plant's energy usage statistics.
        :rtype: dict
        :raises FusionSolarException: If the API response indicates failure or lacks the
                                       required data.
        """
        # Set the query time to the start of the current day if not provided
        if not query_time:
            query_time = self._get_day_start_sec()

        response = self._session.get(
            url=f"https://{self._huawei_subdomain}.fusionsolar.huawei.com/rest/pvms/web/station/v1/overview/energy-balance",
            params={
                "stationDn": plant_id,
                "timeDim": 2,
                "queryTime": query_time,
                # NOTE: Ensure timestamp aligns with API expectations (milliseconds or microseconds)
                "timeZone": time_zone,  # Example: 1 for no daylight saving
                "timeZoneStr": time_zone_str,
                "_": round(time.time() * 1000),  # Timestamp for API request tracking
            },
        )
        response.raise_for_status()
        plant_data = response.json()

        if not plant_data.get("success") or "data" not in plant_data:
            raise FusionSolarException(
                f"Failed to retrieve plant statistics for plant ID {plant_id}."
            )

        return plant_data["data"]

    @logged_in
    def get_plant_details(
            self,
            plant_id: str,
    ) -> dict:
        response = self._session.get(
            url=f"https://{self._huawei_subdomain}.fusionsolar.huawei.com/rest/pvms/web/station/v1/overview/station-detail",
            params={
                "stationDn": plant_id,
                "_": round(time.time() * 1000),  # Timestamp for API request tracking
            },
        )
        response.raise_for_status()
        plant_details = response.json()

        if not plant_details.get("success") or "data" not in plant_details:
            raise FusionSolarException(
                f"Failed to retrieve plant statistics for plant ID {plant_id}."
            )

        return plant_details["data"]

    class BatterySignal(Enum):
        """Enumeration of battery signal types."""
        CHARGING_KWH = "30001"
        DISCHARGING_KWH = "30002"
        CHARGE_DISCHARGE_POWER_KW = "30005"
        VOLTAGE = "30006"
        SOC = "30007"

    @logged_in
    def get_battery_day_stats(
            self,
            battery_id: str,
            query_time: int = None,
            signals: List[BatterySignal] = None,
    ) -> dict:
        """
        Retrieves daily statistics for a specified battery.

        This method fetches data for the given battery ID, including state of charge (SOC)
        and charge/discharge power, for the current day or a specified date.

        :param battery_id: The unique identifier of the battery.
        :type battery_id: str
        :param query_time: A timestamp representing the start of the day (00:00:00)
                           for which the data should be retrieved, in milliseconds.
                           Defaults to the current day if not provided.
        :type query_time: int, optional
        :param signals: A list of `BatterySignal` enum members specifying the
                        desired metrics to retrieve. Defaults to SOC and charge/discharge power.
        :type signals: List[BatterySignal], optional
        :return: A dictionary containing the requested battery data.
        :rtype: dict
        :raises ValueError: If `battery_id` is invalid or if `signals` contains invalid entries.
        :raises FusionSolarException: If the API response indicates failure or lacks the
                                       required data.
        """
        # Input validation
        if not battery_id or not isinstance(battery_id, str):
            raise ValueError("Invalid battery_id. It must be a non-empty string.")

        if signals is None:
            signals = [self.BatterySignal.CHARGE_DISCHARGE_POWER_KW, self.BatterySignal.SOC]
        elif not all(isinstance(signal, self.BatterySignal) for signal in signals):
            raise ValueError("Invalid signals. All items must be instances of `BatterySignal`.")

        if query_time is None:
            query_time = self._get_day_start_sec()  # Default to the start of the current day

        current_time = round(time.time() * 1000)  # Current timestamp in milliseconds

        # API request
        response = self._session.get(
            url=f"https://{self._huawei_subdomain}.fusionsolar.huawei.com/rest/pvms/web/device/v1/device-history-data",
            params={
                "signalIds": [signal.value for signal in signals],  # Convert enums to their values
                "deviceDn": battery_id,
                "date": query_time,
                "_": current_time,
            },
        )
        response.raise_for_status()

        battery_data = response.json()

        if not battery_data.get("success") or "data" not in battery_data:
            raise FusionSolarException(
                f"Failed to retrieve battery day stats for battery ID {battery_id}."
            )

        # Map signal IDs to their names
        for key in battery_data["data"]:
            battery_data["data"][key]["name"] = self.BatterySignal(key).name

        return battery_data["data"]

    def get_parsed_plant_flow(self, plant_id: str) -> Tuple[float, float, float, float, float]:
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
