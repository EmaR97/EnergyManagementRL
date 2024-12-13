# weather.py

from datetime import datetime

import openmeteo_requests
import pandas as pd
import requests_cache
from pvlib import location
from retry_requests import retry


def setup_weather_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_weather_data(
        client: openmeteo_requests.Client,
        start_time: str,
        end_time: str,
        latitude: float,
        longitude: float
) -> any:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": datetime.strptime(start_time, '%Y-%m-%d %H:%M').strftime('%Y-%m-%d'),
        "end_date": datetime.strptime(end_time, '%Y-%m-%d %H:%M').strftime('%Y-%m-%d'),
        "minutely_15": ["shortwave_radiation_instant", "diffuse_radiation_instant", "direct_normal_irradiance_instant"],
        "timezone": 'auto'
    }
    return client.weather_api(url, params=params)[0]


def process_weather_data(
        response,
        timezone: str
) -> pd.DataFrame:
    minutely_15 = response.Minutely15()
    weather_data = {
        "date": pd.date_range(
            start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
            end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=minutely_15.Interval()),
            inclusive="left"
        ),
        "ghi": minutely_15.Variables(0).ValuesAsNumpy(),
        "dhi": minutely_15.Variables(1).ValuesAsNumpy(),
        "dni": minutely_15.Variables(2).ValuesAsNumpy()
    }
    return pd.DataFrame(weather_data).set_index('date').tz_convert(timezone)


def get_weather_data_clearsky(end_time: str, loc: location, timezone: str, start_time: str):
    times = pd.date_range(start_time, end_time, freq='15min')#.tz_localize(timezone)
    weather = loc.get_clearsky(times)
    return weather


def get_weather_data_openmeteo(
        start_time: str,
        end_time: str,
        latitude: float,
        longitude: float,
        timezone: str
):
    weather_client = setup_weather_client()
    weather_response = fetch_weather_data(
        weather_client, start_time, end_time, latitude, longitude
    )
    print(f"Coordinates: {weather_response.Latitude()}°N, {weather_response.Longitude()}°E")
    print(f"Elevation: {weather_response.Elevation()} m asl")
    print(f"Timezone: {weather_response.Timezone()} {weather_response.TimezoneAbbreviation()}")
    return process_weather_data(weather_response, timezone)
