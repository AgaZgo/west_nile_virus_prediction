from typing import Tuple
from loguru import logger
from imblearn.under_sampling import (
    NearMiss, TomekLinks, EditedNearestNeighbours,
    OneSidedSelection, NeighbourhoodCleaningRule
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN

import pandas as pd
import numpy as np


def stratified_undersample(df: pd.DataFrame) -> pd.DataFrame:
    df_0 = df[df.WnvPresent == 0].groupby(
        ['Trap', 'Year', 'Month']).sample(frac=0.15)
    df_1 = df[df.WnvPresent == 1]
    return pd.concat([df_0, df_1])


def resample(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    method: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies resampling to imbalanced training and returns balanced data

    Args:
        features (pd.DataFrame): Dataframe with features
        labels (pd.DataFrame): Series with labels
        method (str): Resampling method

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Balanced features and labels
    """

    # choose undersampling method
    if method == 'NearMiss':
        resample = NearMiss(version=1, n_neighbors=3)
    elif method == "TomekLinks":
        resample = TomekLinks()
    elif method == "EditedNearestNeighbours":
        resample = EditedNearestNeighbours(n_neighbors=3)
    elif method == "OneSidedSelection":
        resample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
    elif method == "NeighbourhoodCleaningRule":
        resample = NeighbourhoodCleaningRule(
            n_neighbors=3, threshold_cleaning=0.5)
    # or oversampling method
    elif method == 'SMOTE':
        resample = SMOTE()
    elif method == 'ADASYN':
        resample = ADASYN()
    # or combined method
    elif method == 'SMOTETomek':
        resample = SMOTETomek()
    elif method == 'SMOTEENN':
        resample = SMOTEENN()

    logger.debug(f'Resampling data using: {method}')
    features, labels = resample.fit_resample(
        features.astype(np.float64),
        labels.astype(np.float64)
    )

    return features, labels
