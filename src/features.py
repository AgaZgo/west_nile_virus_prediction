from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import pandas as pd

from src.config import LAG_RANGE, WINDOW_RANGE
from src.config import NUM_AGG_FEATURES, NUM_WEATHER_FEATURES, SELECTOR


class SpeciesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.enc = None

    def fit(self, df: pd.DataFrame):
        self.enc = OneHotEncoder(sparse_output=False)
        self.enc.fit(df['Species'].to_frame())

        return self

    def transform(self, df: pd.DataFrame):
        df_species = pd.DataFrame(
            self.enc.transform(df['Species'].to_frame()),
            columns=self.enc.get_feature_names_out()
        )
        return pd.concat(
            [
                df.reset_index(drop=True).drop(['Species'], axis=1),
                df_species
            ],
            axis=1
        )


def add_lag_window_to_column_name(
    df: pd.DataFrame,
    lag: int,
    window: int
):
    """Extends column names of a dataframe by appending number of lagged days
    and size of aggregation window

    Args:
        df (pd.DataFrame): dataframe with column names to be updated
        lag (int): number of lagged days
        window (int): window for aggregation function
    """
    df.columns = ['_'.join([c, f'mean_l{lag}_w{window}']) for c in df.columns]


def aggregate_columns_with_lag(
    df: pd.DataFrame,
    lag_range: Tuple[int, int, int],
    window_range: Tuple[int, int, int],
    agg_func: str
) -> pd.DataFrame:
    """Performs an aggregation with moving window with lagging for all columns
    in a dataframe. Aggregation is made for each combination of lag and window
    size within lag and window range.

    Args:
        df (pd.DataFrame): dataframe with columns to aggregate
        lag_range (Tuple[3]): minimal lag, maximal lag, step
        window_range (Tuple[3]): minimal window, maximal window, step
        agg_func (str): aggregation function

    Returns:
        pd.DataFrame: dataframe of aggregated and lagged columns
    """
    df.set_index('Date', inplace=True)
    df_agg = pd.DataFrame(index=df.index)
    for lag in range(lag_range[0], lag_range[1], lag_range[2]):
        for window in range(window_range[0], window_range[1], window_range[2]):
            df_one = df.shift(lag).rolling(window).agg(agg_func)
            add_lag_window_to_column_name(df_one, lag, window)
            df_agg = pd.concat([df_agg, df_one], axis=1).dropna()
    return df_agg


def select_features(features, labels, num_features, model_type):
    if model_type == 'log_regr':
        classifier = LogisticRegression()  # data must be scaled
    elif model_type == 'xgb':
        classifier = XGBClassifier()
    sfs_forward = SequentialFeatureSelector(
        classifier,
        n_features_to_select=num_features,
        direction='forward',
        n_jobs=-1
    ).fit(features, labels)
    return sfs_forward


def select_weather_features(
    df: pd.DataFrame,
    df_weather: pd.DataFrame,
    num_features: int,
    selector: str
) -> list:

    df = pd.merge(df, df_weather.reset_index(), on='Date')
    X_train = df[df_weather.columns]
    y_train = df['WnvPresent']
    sfs_forward = select_features(X_train, y_train, num_features, selector)

    return df_weather.columns[sfs_forward.get_support()].to_list()


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        weather: pd.DataFrame,
        agg_weather: pd.DataFrame,
        num_weather_features: int,
        num_agg_features: int,
        selector: str
    ):
        self.weather = weather
        self.agg_weather = agg_weather
        self.num_weather_features = num_weather_features
        self.num_agg_features = num_agg_features
        self.selector = selector

        self.selected_weather_cols = None
        self.selected_agg_cols = None

    def fit(
        self,
        df: pd.DataFrame
    ):

        self.selected_weather_cols = select_weather_features(
            df,
            self.weather,
            self.num_weather_features,
            self.selector
        )

        self.selected_agg_cols = select_weather_features(
            df,
            self.agg_weather,
            self.num_agg_features,
            self.selector
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data_full = pd.merge(
            pd.merge(df, self.weather.reset_index(), on='Date'),
            self.agg_weather.reset_index(),
            on='Date'
        )
        df = data_full[
            [
                *df.columns.to_list(),
                *self.selected_weather_cols,
                *self.selected_agg_cols
            ]
        ]
        df.drop(['Date', 'Month', 'Trap'], axis=1, inplace=True)
        return df


def get_features(data: dict) -> Tuple[pd.DataFrame]:

    df_agg = aggregate_columns_with_lag(
        data['weather'],
        lag_range=LAG_RANGE,
        window_range=WINDOW_RANGE,
        agg_func='mean'
    )

    feature_selector = FeatureSelector(
        data['weather'],
        df_agg,
        NUM_WEATHER_FEATURES,
        NUM_AGG_FEATURES,
        SELECTOR
    )

    df_train = feature_selector.fit_transform(data['train'])
    df_test = feature_selector.transform(data['test'])

    return df_train, df_test
