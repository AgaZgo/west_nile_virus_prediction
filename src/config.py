# preprocessing
MONTHS = [5, 6, 7, 8, 9]
SPECIES = ['CULEX PIPIENS', 'CULEX PIPIENS/RESTUANS',
           'CULEX RESTUANS'] 
KEEP_MULTIROWS = False
USE_MAX_CATCH_DATE = False

# feature engineering config
TRAIN_COLUMNS = ['Week', 'Dayofyear', 'Latitude', 'Longitude',
                 'trap_max_mos', 'trap_wnv_proba', 'NumRows',
                 # 'lat_bin', 'long_bin',
                 'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
                 'Species_CULEX RESTUANS', 'WnvPresent']

TEST_COLUMNS = ['Id'] + TRAIN_COLUMNS[:-1]

WEATHER_COLUMNS = ['Tmax_1', 'Tmin_1', 'Tavg_1', 'DewPoint_1',
                   'WetBulb_1', 'PrecipTotal_1', 'AvgSpeed_1',
                   'PrecipTotal_2',
                   # 'AvgSpeed_2', 'ResultSpeed_2', 'ResultDir_2',
                   # 'ResultDir_1', 'ResultSpeed_1'
                   ]

AGG_COLS = ['Tmax_1', 'Tmin_1', 'PrecipTotal_1', 'DewPoint_1',
            'AvgSpeed_1', 'PrecipTotal_2', 'Tavg_1'
            # WetBulb_1'
            ]

LAG_LIST = [1, 8, 15]
WINDOW_LIST = [7]
AGG_LIST = ['max', 'mean']

NUM_WEATHER_FEATURES = 0   # number of weather features to select
NUM_AGG_FEATURES = 0  # number of aggregated weather features to select
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
