from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline
from loguru import logger

import pandas as pd

from src.config import AGG_COLS, LAG_LIST, WINDOW_LIST, AGG_LIST
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


class TrapFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.trap_max_mos = None
        self.trap_wnv_proba = None

    def fit(self, df):
        self.trap_max_mos = df.groupby(['Date', 'Trap'])['NumMosquitos'].sum(
            ).reset_index().groupby(['Trap'])['NumMosquitos'].max()
        self.trap_wnv_proba = df.groupby(['Date', 'Trap'])['WnvPresent'].max(
            ).reset_index().groupby(['Trap'])['WnvPresent'].mean()
        return self
        
    def transform(self, df):
        df['trap_max_mos'] = df.Trap.map(
            self.trap_max_mos.to_dict()).fillna(self.trap_max_mos.median())
        df['trap_wnv_proba'] = df.Trap.map(
            self.trap_wnv_proba.to_dict()).fillna(self.trap_wnv_proba.median())
        return df


def add_lag_window_to_column_name(
    df: pd.DataFrame,
    lag: int,
    window: int,
    agg_f: str
):
    """Extends column names of a dataframe by appending number of lagged days
    and size of aggregation window

    Args:
        df (pd.DataFrame): dataframe with column names to be updated
        lag (int): number of lagged days
        window (int): window for aggregation function
    """
    df.columns = ['_'.join([c, f'{agg_f}_l{lag}_w{window}'])
                  for c in df.columns]


def aggregate_columns_with_lag(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    windows: List[int],
    agg_func: List[str]
) -> pd.DataFrame:
    """Performs an aggregation with moving window with lagging for all columns
    in a dataframe. Aggregation is made for each combination of lag and window
    size within lag and window range.

    Args:
        df (pd.DataFrame): dataframe with columns to aggregate
        lags (List[int]): list of lag sizes to apply
        windows (List[int]): list of window sizes to apply
        agg_func (str): list of aggregation functions

    Returns:
        pd.DataFrame: dataframe of aggregated and lagged columns
    """
    df.set_index('Date', inplace=True)
    df = df[AGG_COLS]
    df_agg = pd.DataFrame(index=df.index)
    for lag in lags:
        for window in windows:
            for agg_f in agg_func:
                df_one = df.shift(lag).rolling(window).agg(agg_f)
                add_lag_window_to_column_name(df_one, lag, window, agg_f)
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
    # add multirows count
    multirows_counter = FunctionTransformer(add_num_multirows)
    # encode 'Species'
    species_oh_encoder = SpeciesEncoder()
    # extract trap features
    trap_feat_extractor = TrapFeatureExtractor()

    feature_pipeline = make_pipeline(
        multirows_counter,
        species_oh_encoder,
        trap_feat_extractor
    )
    data['train'] = feature_pipeline.fit_transform(data['train'])
    data['test'] = feature_pipeline.transform(data['test'])

    # get aggregated and lagged weather features
    logger.debug('Aggregating weather with lag...')
    df_agg = aggregate_columns_with_lag(
        data['weather'],
        columns=AGG_COLS,
        lags=LAG_LIST,
        windows=WINDOW_LIST,
        agg_func=AGG_LIST
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

    logger.info('Features selection finished.')
    return df_train, df_test


def add_num_multirows(df: pd.DataFrame) -> pd.DataFrame:

    multirows = df.groupby(
        ['Date', 'Trap', 'Species']
    ).agg(NumRows=('Latitude', 'count')).reset_index()
    df = df.merge(multirows, on=['Date', 'Trap', 'Species'], how='left')

    return df
