import pandas as pd

from src.paths import PREPROCESSED_DATA_DIR
from src.config import UNDERSAMPLE, LAG_RANGE, WINDOW_RANGE
from src.config import NUM_WEATHER_FEATURES, NUM_AGG_FEATURES, SELECTOR
from src.training import undersample
from src.data import aggregate_columns_with_lag, FeatureSelector


df_train = pd.read_pickle(PREPROCESSED_DATA_DIR / 'preprocessed_train.pkl')
df_test = pd.read_pickle(PREPROCESSED_DATA_DIR / 'preprocessed_test.pkl')
df_weather = pd.read_pickle(PREPROCESSED_DATA_DIR / 'clean_weather.pkl')

df_agg = aggregate_columns_with_lag(
    df_weather,
    lag_range=LAG_RANGE,
    window_range=WINDOW_RANGE,
    agg_func='mean'
)

if UNDERSAMPLE:
    data_train = undersample(df_train)
else:
    data_train = df_train.copy()

feature_selector = FeatureSelector(
    df_weather,
    df_agg,
    NUM_WEATHER_FEATURES,
    NUM_AGG_FEATURES,
    SELECTOR
)
data_train = feature_selector.fit_transform(data_train)
data_test = feature_selector.transform(df_test)

data_train.to_pickle(PREPROCESSED_DATA_DIR / 'data_train.pkl')
data_test.to_pickle(PREPROCESSED_DATA_DIR / 'data_test.pkl')
