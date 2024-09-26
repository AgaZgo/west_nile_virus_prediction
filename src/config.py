# feature engineering config
LAG_LIST = [1, 3, 7, 14]
WINDOW_LIST = [3, 7, 10]
NUM_WEATHER_FEATURES = 20    # number of weather features to select
NUM_AGG_FEATURES = 10    # number of aggregated weather features to select
FEATURE_SELECTOR = 'xgb'  # no feature scaling required

# training config

# Choose resampling method from:
# ['stratified_undersample', 'NearMiss', 'TomekLinks',
# 'EditedNearestNeighbours', 'OneSidedSelection', 'NeighbourhoodCleaningRule',
# 'SMOTE', 'ADASYN', 'SMOTETomek', 'SMOTEENN']
RESAMPLE_METHOD = 'stratified_undersample'

MODEL = 'lgbm'
N_TRIALS = 10

# model parameters
N_ESTIMATORS = [50, 60, 100, 150, 200]
MAX_DEPTH = (3, 12)
MIN_CHILD_WEIGHT = [1, 3, 5, 7]
SUBSAMPLE = (0.6, 1.0)
LEARNING_RATE = (1e-2, 0.1)
LAMBDA = (0.01, 1)

MLFLOW_URI = "http://127.0.0.1:8090"
EXPERIMENT_NAME = "West Nile Virus"
