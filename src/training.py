from typing import Tuple
from loguru import logger
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold

import pandas as pd
import optuna

from src.resampling import stratified_undersample, resample
from src.config import RESAMPLE, N_TRIALS


def get_training_data(
    df_train: pd.DataFrame,
    method: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Performs resampling of training data.
    Removes redundant columns

    Args:
        df_train (pd.DataFrame): training data
        method (str): resampling method

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and labels ready for training
                                        a model
    """

    if method == 'stratified_undersample':
        df_train = stratified_undersample(df_train)

        labels = df_train.pop('WnvPresent')
        features = df_train.drop(
            ['Date', 'Month', 'Trap', 'NumMosquitos'],
            axis=1
        )
    else:
        labels = df_train.pop('WnvPresent')
        features = df_train.drop(
            ['Date', 'Month', 'Trap', 'NumMosquitos'],
            axis=1
        )
        features, labels = resample(features, labels, method)

    logger.debug(f'Resampled data ratio: \
        {int(labels.sum())}:{int(labels.shape[0]-labels.sum())}')

    return features, labels


def tune_objective(
    trial: optuna.trial.Trial,
    features: pd.DataFrame,
    labels: pd.Series
) -> float:
    """Performs cross-validation of a model and its parameters chosen by
    trial object.

    Args:
        trial (optuna.trial.Trial): optuna Trial object
        features (pd.DataFrame): Training features
        labels (pd.Series): Training labels

    Returns:
        float: Mean roc_auc test score from cv folds
    """

    # choose model
    model = trial.suggest_categorical('model', ['lgbm', 'xgb'])

    # choose model's hyperparameters
    if model == 'lgbm':
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            "min_child_weight": trial.suggest_categorical(
                'min_child_weight',
                [1, 3, 5]
            ),
            "subsample": trial.suggest_float('subsample', 0.5, 1.0),
            "learning_rate": trial.suggest_float(
                'learning_rate', 1e-4, 10, log=True
            ),
            "reg_lambda": trial.suggest_float('reg_lambda', 0.01, 10),
            "scale_pos_weight": 1 if RESAMPLE else 9
        }
        clf = LGBMClassifier(**params, verbose=-1)
    else:
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            "min_child_weight": trial.suggest_categorical(
                'min_child_weight',
                [1, 3, 5]
            ),
            "subsample": trial.suggest_float('subsample', 0.5, 1.0),
            "eta": trial.suggest_float('eta', 1e-4, 10, log=True),
            "lambda": trial.suggest_float('lambda', 0.01, 10),
            "scale_pos_weight": 1 if RESAMPLE else 9
        }
        clf = XGBClassifier(**params)

    # create CV folds
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    # perform cross-validation
    cv_results = cross_validate(
        clf, features, labels, scoring='roc_auc', cv=cv
    )

    return cv_results['test_score'].mean()


def train_best_model(
    best_params: dict,
    features: pd.DataFrame,
    labels: pd.Series
):
    """Takes best model found by optuna through cross-validation and trains
    it on full training dataset

    Args:
        best_params (dict): Dictionary with best parameters chosen by optuna
        features (pd.DataFrame): Training features
        labels (pd.Series): Training labels

    Returns:
        Trained model compatible with sklearn API
    """
    model = best_params.pop('model', None)

    if model == 'lgbm':
        clf = LGBMClassifier(**best_params)
    elif model == 'xgb':
        clf = XGBClassifier(**best_params)

    clf.fit(features, labels)

    logger.info('Best model fitted.')

    return clf


def train_with_hyperparams_tuning(
    features: pd.DataFrame,
    labels: pd.Series
):
    """Performs model selection with hyperparameters tuning

    Args:
        features (pd.DataFrame): Training features
        labels (pd.Series): Training labels

    Returns:
        Trained model compatible with sklearn API
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: tune_objective(trial, features, labels),
        n_trials=N_TRIALS
    )

    best_model = train_best_model(study.best_params, features, labels)
    return best_model
