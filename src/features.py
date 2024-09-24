from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from loguru import logger

import pandas as pd

from src.config import LAG_RANGE, WINDOW_RANGE
from src.config import NUM_AGG_FEATURES, NUM_WEATHER_FEATURES, FEATURE_SELECTOR
from src.feature_selector import FeatureSelector


class SpeciesEncoder(BaseEstimator, TransformerMixin):
    """Class to apply one-hot encoding to 'Species' columns and replace it
    with new one-hot columns"""

    def __init__(self):
        self.enc = OneHotEncoder(sparse_output=False)

    def fit(self, df: pd.DataFrame):
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


def get_features(data: dict) -> Tuple[pd.DataFrame]:
    """Performes feature engineering:
        - one-hot encoding of species column
        - generates aggregated weather features with lag
        - selects subset of most promising features for modeling

    Args:
        data (dict): Dictionary of clean and preprocessed data:
                        {'train': pd.DataFrame, 'test': pd.DataFrame,
                        'weather': pd.DataFrame}

    Returns:
        Tuple[pd.DataFrame]: Tuple of train and test dataframe
    """

    # encode 'Species'
    logger.debug('Encoding species...')
    species_oh_encoder = SpeciesEncoder()
    data['train'] = species_oh_encoder.fit_transform(data['train'])
    data['test'] = species_oh_encoder.transform(data['test'])
    logger.info('Species encoded')

    # get aggregated and lagged weather features
    logger.debug('Aggregating weather with lag...')
    df_agg = aggregate_columns_with_lag(
        data['weather'],
        lag_range=LAG_RANGE,
        window_range=WINDOW_RANGE,
        agg_func='mean'
    )
    logger.info('Weather aggregated and lagged.')

    # build feature selector
    feature_selector = FeatureSelector(
        data['weather'],
        df_agg,
        NUM_WEATHER_FEATURES,
        NUM_AGG_FEATURES,
        FEATURE_SELECTOR
    )

    # select features from train and test data
    df_train = feature_selector.fit_transform(data['train'])
    df_test = feature_selector.transform(data['test'])
    logger.info(
        f'Features selection finished with {df_train.columns.to_list()}')
    return df_train, df_test
