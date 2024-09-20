from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold

import pandas as pd
import optuna

from src.resampling import random_undersample, near_miss
from src.config import UNDERSAMPLE


def get_training_data(
    df_train: pd.DataFrame,
    method: str
) -> pd.DataFrame:

    labels = df_train.pop('WnvPresent')
    features = df_train.drop(['Date', 'Month', 'Trap'], axis=1)

    if method == 'random_undersample':
        features, labels = random_undersample(features, labels)
    elif method == 'nearmiss':
        features, labels = near_miss(features, labels)

    return features, labels


def objective(trial, features, labels):

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
    res = cross_validate(clf, features, labels, scoring='roc_auc', cv=cv)
    return res['test_score'].mean()


def train_best(best_params, features, labels):
    model = best_params.pop('model', None)

    if model == 'lgbm':
        clf = LGBMClassifier(**best_params)
    else:
        clf = XGBClassifier(**best_params)

    clf.fit(features, labels)

    return clf


def train(
    features: pd.DataFrame,
    labels: pd.Series
):
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, features, labels),
        n_trials=4
    )

    best_model = train_best(study.best_params, features, labels)
    return best_model
