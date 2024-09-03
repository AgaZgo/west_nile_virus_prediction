import pandas as pd

from src.paths import PREPROCESSED_DATA_DIR
from src.config import UNDERSAMPLE
from src.training import undersample
from src.data import aggregate_columns_with_lag


df_train = pd.read_pickle(PREPROCESSED_DATA_DIR / 'preprocessed_train.pkl')
df_test = pd.read_pickle(PREPROCESSED_DATA_DIR / 'preprocessed_test.pkl')
df_weather = pd.read_pickle(PREPROCESSED_DATA_DIR / 'clean_weather.pkl')

df_agg = aggregate_columns_with_lag(
    df_weather,
    lag_range=(1, 14, 3),
    window_range=(1, 11, 3),
    agg_func='mean'
)

if UNDERSAMPLE:
    data_train = undersample(df_train)

data_full = pd.merge(
    pd.merge(data_train, df_weather.reset_index(), on='Date'),
    df_agg.reset_index(),
    on='Date'
)
