from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from energymanagementrl.pipeline.config import get_plant_config


class TestFusionSolarClient:
    def test_client_imports(self):
        from energymanagementrl.fusion_solar_connector import FusionSolarClientParsed, BatteryWorkingMode

        assert FusionSolarClientParsed is not None
        assert BatteryWorkingMode is not None

    def test_battery_working_mode_values(self):
        from energymanagementrl.fusion_solar_connector import BatteryWorkingMode

        assert BatteryWorkingMode.MAXIMUM_SELF_CONSUMPTION.value == 2
        assert BatteryWorkingMode.FULLY_FEED_TO_GRID.value == 4

    def test_config_builds_plant_config(self, sample_config, monkeypatch):
        monkeypatch.setenv("LAT", "45.0")
        monkeypatch.setenv("LON", "9.0")
        plant = get_plant_config(sample_config)
        system = plant.setup_pv_system()
        assert system is not None
        assert len(system.arrays) == 2


def _make_day_df(timestamps, soc_values, empty=False):
    """Build a raw day DataFrame as fetch_statistics would return it."""
    cols = ['productPower', 'usePower', 'selfUsePower', 'chargePower', 'dischargePower', 'SOC', 'GRID_VOLTAGE']
    data = {}
    for c in cols:
        if c == 'SOC':
            data[c] = soc_values
        elif empty:
            data[c] = ['--'] * len(timestamps)
        else:
            data[c] = np.random.uniform(0, 5, len(timestamps))
    return pd.DataFrame(data, index=timestamps)


def _make_client_with_mocked_fetch(fake_fetch):
    """Create a FusionSolarClientParsed with fetch_statistics replaced."""
    from energymanagementrl.fusion_solar_connector.parsed_client import FusionSolarClientParsed

    client = MagicMock()
    client.fetch_statistics = fake_fetch
    client.get_plant_history = FusionSolarClientParsed.get_plant_history.__get__(client, type(client))
    return client


class TestGetPlantHistoryEarlyStop:
    """Tests that get_plant_history iterates newest→oldest and stops after 7 consecutive empty days."""

    def test_stops_after_7_consecutive_empty_days(self):
        """Newest 7 days are empty → should stop, fetch exactly 7, return empty result."""
        call_count = [0]

        def fake_fetch(battery_id, inverter_id, plant_id, timestamp, tz):
            call_count[0] += 1
            ts = pd.to_datetime(timestamp, unit='ms')
            timestamps = pd.date_range(ts, periods=10, freq='5min')
            soc = np.random.uniform(20, 80, 10)
            # First 7 calls (newest days) are empty; rest would be real
            is_empty = call_count[0] <= 7
            return _make_day_df(timestamps, soc, empty=is_empty)

        client = _make_client_with_mocked_fetch(fake_fetch)

        start = pd.Timestamp("2025-03-17")
        end = pd.Timestamp("2025-03-31")
        unformatted, final = client.get_plant_history(
            start, end, "bat1", "inv1", "plant1", "Europe/Rome"
        )

        assert call_count[0] == 7
        assert len(final) == 0

    def test_stops_early_collects_real_days(self):
        """3 empty newest days, then 7 real → fetches all 10, keeps 7 real."""
        call_count = [0]

        def fake_fetch(battery_id, inverter_id, plant_id, timestamp, tz):
            call_count[0] += 1
            ts = pd.to_datetime(timestamp, unit='ms')
            timestamps = pd.date_range(ts, periods=10, freq='5min')
            soc = np.random.uniform(20, 80, 10)
            is_empty = call_count[0] <= 3
            return _make_day_df(timestamps, soc, empty=is_empty)

        client = _make_client_with_mocked_fetch(fake_fetch)

        start = pd.Timestamp("2025-03-21")
        end = pd.Timestamp("2025-03-31")
        unformatted, final = client.get_plant_history(
            start, end, "bat1", "inv1", "plant1", "Europe/Rome"
        )

        assert call_count[0] == 10
        assert len(final) > 0

    def test_single_empty_day_does_not_stop(self):
        """1 empty day among 5 total → streak never reaches 7, all fetched."""
        call_count = [0]

        def fake_fetch(battery_id, inverter_id, plant_id, timestamp, tz):
            call_count[0] += 1
            ts = pd.to_datetime(timestamp, unit='ms')
            timestamps = pd.date_range(ts, periods=10, freq='5min')
            soc = np.random.uniform(20, 80, 10)
            is_empty = call_count[0] == 1  # only the very first (newest) day
            return _make_day_df(timestamps, soc, empty=is_empty)

        client = _make_client_with_mocked_fetch(fake_fetch)

        start = pd.Timestamp("2025-03-26")
        end = pd.Timestamp("2025-03-31")
        unformatted, final = client.get_plant_history(
            start, end, "bat1", "inv1", "plant1", "Europe/Rome"
        )

        assert call_count[0] == 5
        assert len(final) > 0

    def test_empty_streak_resets_on_real_day(self):
        """Pattern: empty, empty, real, empty, empty, real, then 4 real → streak resets, all fetched."""
        call_count = [0]
        # Iteration is newest→oldest. Pattern per call: E,E,R,E,E,R,R,R,R
        is_empty_sequence = [True, True, False, True, True, False, False, False, False]

        def fake_fetch(battery_id, inverter_id, plant_id, timestamp, tz):
            idx = call_count[0]
            call_count[0] += 1
            ts = pd.to_datetime(timestamp, unit='ms')
            timestamps = pd.date_range(ts, periods=10, freq='5min')
            soc = np.random.uniform(20, 80, 10)
            is_empty = is_empty_sequence[idx] if idx < len(is_empty_sequence) else False
            return _make_day_df(timestamps, soc, empty=is_empty)

        client = _make_client_with_mocked_fetch(fake_fetch)

        start = pd.Timestamp("2025-03-22")
        end = pd.Timestamp("2025-03-31")
        unformatted, final = client.get_plant_history(
            start, end, "bat1", "inv1", "plant1", "Europe/Rome"
        )

        # All 9 fetched, streak never reaches 7
        assert call_count[0] == 9
        assert len(final) > 0
