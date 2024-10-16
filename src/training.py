from typing import Tuple
from loguru import logger
# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
import optuna
import mlflow

from src.pipeline import get_pipeline
from src.mlflow_utils import generate_run_name, log_config_to_mlflow
from src.config import (
    MODEL, N_TRIALS, MLFLOW_URI, EXPERIMENT_NAME,
    N_ESTIMATORS, MAX_DEPTH, MIN_CHILD_WEIGHT, SUBSAMPLE,
    LEARNING_RATE, LAMBDA
)


mlflow.set_tracking_uri(uri=MLFLOW_URI)


def get_training_data(
    df_train: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Decouples features and labels

    Args:
        df_train (pd.DataFrame): training data

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and labels ready for training
                                        a model
    """
    columns_to_drop = ['Date'] #, 'Month', 'Trap', 'NumMosquitos', 'Year',
                       #'Dayofyear', 'Dayofweek']
    # breakpoint()
    # df_train = df_train[df_train.Year.isin([2009, 2011])]
    labels = df_train.pop('WnvPresent').astype(np.float64)
    features = df_train.drop(columns_to_drop, axis=1).astype(np.float64)
    
    # features = df_train.astype(np.float64)

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
            'n_estimators': trial.suggest_categorical(
                'n_estimators', N_ESTIMATORS),
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
    elif MODEL == 'lr':
        params = {
            'C': trial.suggest_float('C', 0.1, 10, log=True)
        }
        clf = LogisticRegression(**params)

    pipeline = get_pipeline(clf=clf)

    # create CV folds
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    # perform cross-validation
    cv_results = cross_validate(
        pipeline, features, labels, scoring='roc_auc', cv=cv
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
    elif MODEL == 'lr':
        clf = LogisticRegression(**best_params)

    pipeline = get_pipeline(clf=clf)

    pipeline.fit(features, labels)
    logger.info('Best model fitted. Loading to MLflow...')

    mlflow.start_run(run_name=generate_run_name())
    log_config_to_mlflow()

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
        sk_model=pipeline,
        artifact_path="wnv",
        signature=signature,
        input_example=features,
        registered_model_name=f"{MODEL}-for-WNV",
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
