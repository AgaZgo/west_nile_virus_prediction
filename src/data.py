from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from typing import Tuple

import pandas as pd


def split_date(df: pd.DataFrame):
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


def filter_and_encode_species(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Removes rows with species for which test results never came back
    positive (virus never present). Encodes species by integers

    Args:
        df (pd.DataFrame): dataframe with columns "Species" and 'WnvPresent'

    Returns:
        pd.DataFrame: dataframe with species tested positive for wnv
        dict: species to index
    """
    virus_per_species = df.groupby('Species')['WnvPresent'].sum()
    positive_species = virus_per_species[virus_per_species > 0].index.to_list()
    species2index = {s: i for i, s in enumerate(positive_species)}
    df['Species'] = df['Species'].map(species2index)
    return df.dropna(), species2index


def filter_months(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with months with number of positive tests less than 3.

    Args:
        df (pd.DataFrame): dataframe with columns "Month" and 'WnvPresent

    Returns:
        pd.DataFrame: dataframe trimmed to months
            with significant risk of virus presence
    """
    virus_per_month = df.groupby('Month')['WnvPresent'].sum()
    positive_months = virus_per_month[virus_per_month > 2].index
    df = df[df['Month'].isin(positive_months)]
    return df


def filter_traps(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with traps which never caught infected mosquito.

    Args:
        df (pd.DataFrame): dataframe with columns 'Trap' and 'WnvPresent'

    Returns:
        pd.DataFrame: dataframe with traps in which infected mosquitos 
            were caught
    """
    virus_per_trap = df.groupby('Trap')['WnvPresent'].sum()
    positive_traps = virus_per_trap[virus_per_trap > 0].index
    df = df[df['Trap'].isin(positive_traps)]
    return df


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
    if model_type == 'lin_regr':
        classifier = LinearRegression()
        sfs_forward = SequentialFeatureSelector(
            classifier,
            n_features_to_select=num_features,
            direction='forward',
            n_jobs=-1
        ).fit(features, labels)
    return sfs_forward
