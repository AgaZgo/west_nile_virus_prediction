import pandas as pd


def flatten_index(df):
    df.columns = [c[0]+'_'+c[1] for c in df.columns.to_flat_index()]


def add_lag_window(df, lag, window):
    df.columns = ['_'.join([c, str(lag), str(window)]) for c in df.columns]


def aggregate_columns(df, lag_range, window_range, agg_func):
    df_agg = pd.DataFrame(index=df.index)
    for lag in range(lag_range[0], lag_range[1], lag_range[2]):
        for window in range(window_range[0], window_range[1], window_range[2]):
            df_one = df.shift(lag).rolling(window).agg(agg_func)
            flatten_index(df_one)
            add_lag_window(df_one, lag, window)
            df_agg = pd.concat([df_agg, df_one], axis=1).dropna()
    return df_agg


