import pandas as pd

from src.config import RANDOM_SEED


def undersample(df: pd.DataFrame) -> pd.DataFrame:
    df_0 = df[df.WnvPresent == 0].groupby(
        ['Trap', 'Year', 'Month']).sample(frac=0.15, random_state=RANDOM_SEED)
    df_1 = df[df.WnvPresent == 1]
    return pd.concat([df_0, df_1])
