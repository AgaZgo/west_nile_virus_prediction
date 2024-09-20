RANDOM_SEED = 42

# feature engineering config
LAG_RANGE = (1, 2, 3)
WINDOW_RANGE = (3, 4, 3)
NUM_WEATHER_FEATURES = 1    # number of weather features to select
NUM_AGG_FEATURES = 1    # number of aggregated weather features to select
NUM_FEATURES = 5   # total number of features to select
FEATURE_SELECTOR = 'xgb'  # no feature scaling required

# training config
UNDERSAMPLE = False  # undersampling negative class
RESAMPLE_METHOD = 'nearmiss'