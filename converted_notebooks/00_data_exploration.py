# %%
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from nixtla import NixtlaClient
from scipy.ndimage import gaussian_filter1d
import pandas as pd

complete_series_csv = '../data/complete_series.csv'
df = pd.read_csv(complete_series_csv, parse_dates=['timestamp'])
# %%
length = 268
sigma = 12  # Adjust to control smoothing intensity
df['smoothed'][:length] = gaussian_filter1d(df['consumption_w'][:length], sigma=sigma)

data = df[:length]
plt.figure(figsize=(10, 6))
plt.plot(data['timestamp'], data['consumption_w'], label='Original Series', alpha=0.7, linewidth=2)
plt.plot(data['timestamp'], data['smoothed'], label='Smoothed Series (Moving Average)', color='orange', linewidth=2)
plt.title('Original vs Smoothed Time Series', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Consumption', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
# %%
load_dotenv()
api_key = os.environ.get("NIXIA_API_KEY")

nixtla_client = NixtlaClient(
    api_key=api_key
)
nixtla_client.validate_api_key()
nixtla_client.plot(df, time_col='timestamp', target_col='smoothed')
# %%
# Resample to 1-hour frequency, aggregating with mean (or another aggregation method)
df_resampled = df.set_index('timestamp').resample('1H').mean().reset_index()
# %%
c_l = 24 * 7
h = 24
level = [90, 99]
# Use the resampled dataframe for forecasting
timegpt_fcst_df = nixtla_client.forecast(
    df=df_resampled[:c_l],
    h=h,
    time_col='timestamp',
    target_col='smoothed',
    X_df=df_resampled[c_l:c_l + h][['timestamp', 'production_w']],
    model='timegpt-1',
    # quantiles=quantiles, 
    add_history=True,
    finetune_steps=10,
    finetune_depth=5,
    level=level,
)

timegpt_fcst_df['TimeGPT'] = gaussian_filter1d(timegpt_fcst_df['TimeGPT'], sigma=2)
for l in level:
    for a in ['lo', 'hi']:
        timegpt_fcst_df[f'TimeGPT-{a}-{l}'] = gaussian_filter1d(timegpt_fcst_df[f'TimeGPT-{a}-{l}'], sigma=2)
# Plot the original data with forecasts
nixtla_client.plot(
    df_resampled[:c_l + h],
    timegpt_fcst_df,
    time_col='timestamp',
    target_col='smoothed',
    level=level,
)
# %%



