from typing import Tuple
from imblearn.under_sampling import (
    NearMiss, TomekLinks, EditedNearestNeighbours,
    OneSidedSelection, NeighbourhoodCleaningRule
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN

import pandas as pd
import numpy as np

from src.config import RANDOM_SEED


def random_undersample(df: pd.DataFrame) -> pd.DataFrame:
    df_0 = df[df.WnvPresent == 0].groupby(
        ['Trap', 'Year', 'Month']).sample(frac=0.15, random_state=RANDOM_SEED)
    df_1 = df[df.WnvPresent == 1]
    return pd.concat([df_0, df_1])


def resample(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    method: str
) -> Tuple[pd.DataFrame, pd.Series]:

    # undersampling methods
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
    # oversampling methods
    elif method == 'SMOTE':
        resample = SMOTE()
    elif method == 'ADASYN':
        resample = ADASYN()
    # combined methods
    elif method == 'SMOTETomek':
        resample = SMOTETomek()
    elif method == 'SMOTEENN':
        resample = SMOTEENN()

    features, labels = resample.fit_resample(
        features.astype(np.float64),
        labels.astype(np.float64)
    )

    return features, labels
