from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple
from xgboost import XGBClassifier

import pandas as pd


def split_date(df: pd.DataFrame) -> pd.DataFrame:
    """Splits 'Date' column into seperate columns for
    month, year, week, day of a year

    Args:
        df (pd.DataFrame): data frame
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Week'] = df.Date.dt.isocalendar().week
    df['Dayofyear'] = df['Date'].dt.dayofyear
    return df


class RowFilterTransformer(BaseEstimator, TransformerMixin):
    """Transformer to drop rows with months, species and traps for which
    wnv presence was detected less than 3 times"""

    def __init__(self, columns=['Species', 'Month', 'Trap']):

        self.columns = columns
        self.positive = dict()

    def fit(self, df: pd.DataFrame):

        for col in self.columns:
            virus_detected_cnt = df.groupby(col)['WnvPresent'].sum()
            self.positive[col] = virus_detected_cnt[
                virus_detected_cnt > 2
            ].index.to_list()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in self.columns:
            df = df[df[col].isin(self.positive[col])]

        return df


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


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['Address', 'Block', 'Street', 'AddressNumberAndStreet',
                       'AddressAccuracy']

    if 'NumMosquitos' in df.columns:
        columns_to_drop.append('NumMosquitos')

    return df.drop(columns_to_drop, axis=1)


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
    df_agg = pd.DataFrame(index=df.index)
    for lag in range(lag_range[0], lag_range[1], lag_range[2]):
        for window in range(window_range[0], window_range[1], window_range[2]):
            df_one = df.shift(lag).rolling(window).agg(agg_func)
            add_lag_window_to_column_name(df_one, lag, window)
            df_agg = pd.concat([df_agg, df_one], axis=1).dropna()
    return df_agg


def select_features(features, labels, num_features, model_type):
    if model_type == 'log_regr':
        classifier = LogisticRegression()  #data must be scaled
    elif model_type == 'xgb':
        classifier = XGBClassifier()
    sfs_forward = SequentialFeatureSelector(
        classifier,
        n_features_to_select=num_features,
        direction='forward',
        n_jobs=-1
    ).fit(features, labels)
    return sfs_forward
