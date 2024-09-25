from typing import Tuple
from loguru import logger
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from mlflow.models import infer_signature

import pandas as pd
import optuna
import mlflow

from src.resampling import stratified_undersample, resample
from src.config import (
    MODEL, N_TRIALS, MLFLOW_URI, EXPERIMENT_NAME,
    MAX_DEPTH, MIN_CHILD_WEIGHT, SUBSAMPLE,
    LEARNING_RATE, LAMBDA
)


mlflow.set_tracking_uri(uri=MLFLOW_URI)


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
    columns_to_drop = ['Date', 'Month', 'Trap', 'NumMosquitos', 'MaxCatchDate',
                       'Year', 'Dayofyear', 'Dayofweek']

    if method == 'stratified_undersample':
        df_train = stratified_undersample(df_train)

        labels = df_train.pop('WnvPresent')
        features = df_train.drop(columns_to_drop, axis=1)
    else:
        labels = df_train.pop('WnvPresent')
        features = df_train.drop(columns_to_drop, axis=1)
        features, labels = resample(features, labels, method)

    logger.debug(f'Resampled data ratio: \
        {int(labels.sum())}:{int(labels.shape[0]-labels.sum())}')

    logger.debug(f"Training with features {features.columns.to_list()}")
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

    # choose model's hyperparameters
    if MODEL == 'lgbm':
        params = {
            'max_depth': trial.suggest_int(
                'max_depth', MAX_DEPTH[0], MAX_DEPTH[1]),
            "min_child_weight": trial.suggest_categorical(
                'min_child_weight', MIN_CHILD_WEIGHT),
            "subsample": trial.suggest_float(
                'subsample', SUBSAMPLE[0], SUBSAMPLE[1]),
            "learning_rate": trial.suggest_float(
                'learning_rate', LEARNING_RATE[0], LEARNING_RATE[1], log=True),
            "reg_lambda": trial.suggest_float(
                'reg_lambda', LAMBDA[0], LAMBDA[1])
        }
        clf = LGBMClassifier(**params, verbose=-1)
    else:
        params = {
            'max_depth': trial.suggest_int(
                'max_depth', MAX_DEPTH[0], MAX_DEPTH[1]),
            "min_child_weight": trial.suggest_categorical(
                'min_child_weight', MIN_CHILD_WEIGHT),
            "subsample": trial.suggest_float(
                'subsample', SUBSAMPLE[0], SUBSAMPLE[1]),
            "eta": trial.suggest_float(
                'eta', LEARNING_RATE[0], LEARNING_RATE[1], log=True),
            "lambda": trial.suggest_float(
                'lambda', LAMBDA[0], LAMBDA[1])
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
    study: optuna.study.study.Study,
    features: pd.DataFrame,
    labels: pd.Series
):
    """Takes best model found by optuna through cross-validation and trains
    it on full training dataset

    Args:
        study (optuna.study.study.Study): optuna Study object containing tuning
                                          expirement results including
                                          'best_params' and 'best_value'
                                          i.e. parameters for best found model
                                          and scoting metric it achieved
        features (pd.DataFrame): Training features
        labels (pd.Series): Training labels

    Returns:
        Trained model compatible with sklearn API
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_params = study.best_params.copy()
    best_score = study.best_value

    if MODEL == 'lgbm':
        clf = LGBMClassifier(**best_params)
    elif MODEL == 'xgb':
        clf = XGBClassifier(**best_params)

    clf.fit(features, labels)
    logger.info('Best model fitted. Loading to MLflow...')

    mlflow.start_run()
    # Log the hyperparameters
    mlflow.log_params(best_params)

    # Log the metric
    mlflow.log_metric("roc_auc_cv", best_score)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag(
        "Training Info",
        f"Basic {MODEL} model for WNV prediction"
    )

    signature = infer_signature(features,
                                clf.predict_proba(features)[:, 1])

    model_info = mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="wnv",
        signature=signature,
        input_example=features,
        registered_model_name="tracking-quickstart",
    )

    loaded_model = mlflow.sklearn.load_model(model_info.model_uri)
    logger.info('Model loaded to MLflow')

    return loaded_model


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

    best_model = train_best_model(study, features, labels)
    return best_model
