from datetime import datetime

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


class OpenMeteoClient:
    def __init__(self, cache_expiry: int = 3600, retries: int = 5, backoff_factor: float = 0.2):
        self.client = self._setup_weather_client(cache_expiry, retries, backoff_factor)

    @staticmethod
    def _setup_weather_client(cache_expiry: int, retries: int, backoff_factor: float) -> openmeteo_requests.Client:
        cache_session = requests_cache.CachedSession('.cache', expire_after=cache_expiry)
        retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
        return openmeteo_requests.Client(session=retry_session)

    def fetch_weather_data(
        self,
        start_time: str,
        end_time: str,
        latitude: float,
        longitude: float,
        forecast: bool = False
    ) -> any:
        url = "https://api.open-meteo.com/v1/forecast" if forecast else "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": datetime.strptime(start_time, '%Y-%m-%d %H:%M').strftime('%Y-%m-%d'),
            "end_date": datetime.strptime(end_time, '%Y-%m-%d %H:%M').strftime('%Y-%m-%d'),
            "minutely_15": ["shortwave_radiation_instant", "diffuse_radiation_instant", "direct_normal_irradiance_instant"],
        }
        return self.client.weather_api(url, params=params)[0]

    @staticmethod
    def process_weather_data(response, start_time: str, end_time: str) -> pd.DataFrame:
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
        df = pd.DataFrame(weather_data).set_index('date').tz_convert(None).resample('5min').interpolate(method='linear')
        filtered_df = df.loc[(df.index >= start_time) & (df.index <= end_time)]
        return filtered_df

    def get_weather_data(
        self,
        start_time: str,
        end_time: str,
        latitude: float,
        longitude: float,
        forecast: bool = False
    ) -> pd.DataFrame:
        weather_response = self.fetch_weather_data(start_time, end_time, latitude, longitude, forecast)
        return self.process_weather_data(weather_response, start_time, end_time)
