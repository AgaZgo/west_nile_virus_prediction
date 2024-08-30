from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

import pandas as pd


def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Week'] = df.Date.dt.isocalendar().week
    df['Dayofyear'] = df['Date'].dt.dayofyear


def filter_and_encode_species(df):
    virus_per_species = df.groupby('Species')['WnvPresent'].sum()
    positive_species = virus_per_species[virus_per_species > 0].index.to_list()
    species2index = {s: i for i, s in enumerate(positive_species)}
    df['Species'] = df['Species'].map(species2index)
    return df.dropna()


def filter_months(df):
    virus_per_month = df.groupby('Month')['WnvPresent'].sum()
    positive_months = virus_per_month[virus_per_month > 2].index
    df = df[df['Month'].isin(positive_months)]
    return df


def filter_traps(df):
    virus_per_trap = df.groupby('Trap')['WnvPresent'].sum()
    positive_traps = virus_per_trap[virus_per_trap > 0].index
    df = df[df['Trap'].isin(positive_traps)]
    return df


def add_lag_window_to_column_name(df, lag, window):
    df.columns = ['_'.join([c, f'mean_l{lag}_w{window}']) for c in df.columns]


def aggregate_columns_with_lag(df, lag_range, window_range, agg_func):
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
