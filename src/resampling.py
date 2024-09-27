import pandas as pd


def stratified_undersample(df: pd.DataFrame) -> pd.DataFrame:
    df_0 = df[df.WnvPresent == 0].groupby(
        ['Trap', 'Year', 'Month']).sample(frac=0.15)
    df_1 = df[df.WnvPresent == 1]
    return pd.concat([df_0, df_1])
