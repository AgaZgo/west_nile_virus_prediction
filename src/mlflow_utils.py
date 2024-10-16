from datetime import datetime

import mlflow

from src.config import (
    LAG_LIST, WINDOW_LIST, NUM_WEATHER_FEATURES, NUM_AGG_FEATURES,
    FEATURE_SELECTOR, MODEL, N_TRIALS, TRAIN_COLUMNS, WEATHER_COLUMNS,
    AGG_COLS, MONTHS
)


def generate_run_name():
    return MODEL + datetime.now().strftime("_%y%m%d_%H%M%S")


def log_config_to_mlflow():
    config_params = {
        'cfg_lag_list': LAG_LIST,
        'cfg_window_list': WINDOW_LIST,
        'cfg_num_weather_features': NUM_WEATHER_FEATURES,
        'cfg_num_agg_features': NUM_AGG_FEATURES,
        'cfg_feature_selector': FEATURE_SELECTOR,
        'cfg_model': MODEL,
        'cfg_n_trails': N_TRIALS,
        'cfg_train_columns': TRAIN_COLUMNS,
        'cfg_weather_columns': WEATHER_COLUMNS,
        'cfg_agg_cols': AGG_COLS,
        'cfg_months': MONTHS
    }

    mlflow.log_params(config_params)
