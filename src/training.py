from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold

import pandas as pd
import pickle
import optuna

from src.paths import MODEL_DIR
from src.config import RANDOM_SEED, UNDERSAMPLE


def random_undersample(df: pd.DataFrame) -> pd.DataFrame:
    df_0 = df[df.WnvPresent == 0].groupby(
        ['Trap', 'Year', 'Month']).sample(frac=0.15, random_state=RANDOM_SEED)
    df_1 = df[df.WnvPresent == 1]
    return pd.concat([df_0, df_1])


def get_training_data(
    df_train: pd.DataFrame,
    method: str
) -> pd.DataFrame:

    if method == 'random_undersample':
        df_train = random_undersample(df_train)

    df_train.drop(['Date', 'Month', 'Trap'], axis=1, inplace=True)

    return df_train


def objective(trial, df_train, labels):

    model = trial.suggest_categorical('model', ['lgbm', 'xgb'])
    if model == 'lgbm':
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            "min_child_weight": trial.suggest_categorical(
                'min_child_weight', [1, 3, 5]
            ),
            "subsample": trial.suggest_float('subsample', 0.5, 1.0),
            "learning_rate": trial.suggest_float(
                'learning_rate', 1e-4, 10, log=True
            ),
            "reg_lambda": trial.suggest_float('reg_lambda', 0.01, 10),
            "scale_pos_weight": 1 if UNDERSAMPLE else 9
        }
        clf = LGBMClassifier(**params, verbose=-1)
    else:
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            "min_child_weight": trial.suggest_categorical(
                'min_child_weight', [1, 3, 5]
            ),
            "subsample": trial.suggest_float('subsample', 0.5, 1.0),
            "eta": trial.suggest_float('eta', 1e-4, 10, log=True),
            "lambda": trial.suggest_float('lambda', 0.01, 10),
            "scale_pos_weight": 1 if UNDERSAMPLE else 9
        }
        clf = XGBClassifier(**params)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    res = cross_validate(clf, df_train, labels, scoring='roc_auc', cv=cv)
    return res['test_score'].mean()


def train_best(best_params, data_train, labels):
    model = best_params.pop('model', None)

    if model == 'lgbm':
        clf = LGBMClassifier(**best_params)
    else:
        clf = XGBClassifier(**best_params)

    clf.fit(data_train, labels)

    return clf


def train(df_train):
    labels = df_train.pop('WnvPresent')
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, df_train, labels),
        n_trials=4
    )

    best_model = train_best(study.best_params, df_train, labels)
    return best_model
