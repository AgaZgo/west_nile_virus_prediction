from typing import Tuple
from imblearn.under_sampling import NearMiss

import pandas as pd

from src.config import RANDOM_SEED


def random_undersample(df: pd.DataFrame) -> pd.DataFrame:
    df_0 = df[df.WnvPresent == 0].groupby(
        ['Trap', 'Year', 'Month']).sample(frac=0.15, random_state=RANDOM_SEED)
    df_1 = df[df.WnvPresent == 1]
    return pd.concat([df_0, df_1])


def near_miss(
    features: pd.DataFrame,
    labels: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    nearmiss = NearMiss(version=1, n_neighbors=3)
    features, labels = nearmiss.fit_resample(features, labels)
    
    return features, labels
    
