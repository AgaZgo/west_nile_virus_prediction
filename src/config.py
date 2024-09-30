# preprocessing
MULTIROWS = True

# feature engineering config
LAG_LIST = [1, 8, 15, 30]
WINDOW_LIST = [7]
AGG_LIST = ['mean', 'max']
NUM_WEATHER_FEATURES = 20   # number of weather features to select
NUM_AGG_FEATURES = 20   # number of aggregated weather features to select
FEATURE_SELECTOR = 'xgb'  # no feature scaling required

# training config
MODEL = 'lgbm'
N_TRIALS = 20

# model parameters
N_ESTIMATORS = [50, 60, 100, 150, 200, 300]
MAX_DEPTH = (3, 12)
MIN_CHILD_WEIGHT = [1, 3, 5, 7]
SUBSAMPLE = (0.6, 1.0)
LEARNING_RATE = (1e-4, 1)
LAMBDA = (0.01, 1)

MLFLOW_URI = "http://127.0.0.1:8090"
EXPERIMENT_NAME = "West Nile Virus"
