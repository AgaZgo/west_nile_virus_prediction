from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline

import pandas as pd
import numpy as np

from src.paths import RAW_DATA_DIR


def read_raw_data() -> dict:
    files = ['train.csv', 'test.csv', 'weather.csv']
    return {
        file_name[:-4]: pd.read_csv(RAW_DATA_DIR / file_name)
        for file_name in files
        }


def build_data_preprocessing_pipeline() -> Pipeline:

    date_transformer = FunctionTransformer(split_date)
    month_species_trap_filter = MonthSpeciesTrapTransformer()
    address_remover = FunctionTransformer(remove_address)

    return make_pipeline(
        date_transformer,
        month_species_trap_filter,
        address_remover,
        memory='cache',
        verbose=True
    )


def clean_weather(df_weather: pd.DataFrame) -> pd.DataFrame:
    columns_to_stay = ['Station', 'Date', 'Tmax', 'Tmin', 'Tavg', 'DewPoint',
                       'WetBulb', 'PrecipTotal']
    df_weather = df_weather[columns_to_stay]

    df_weather.Tavg = np.where(
        df_weather.Tavg == 'M',
        df_weather[['Tmax', 'Tmin']].mean(axis=1),
        df_weather.Tavg
    ).astype('int')

    lr = LinearRegression()

    train_weather = df_weather[df_weather.WetBulb != 'M'][
        ['Tmax', 'Tmin', 'DewPoint', 'WetBulb']
    ]
    test_weather = df_weather[df_weather.WetBulb == 'M'][
        ['Tmax', 'Tmin', 'DewPoint']
    ]
    y_train = train_weather.pop('WetBulb').astype(float)
    X_train = train_weather
    X_test = test_weather

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    missing_index = df_weather[df_weather.WetBulb == 'M'].index
    df_weather.loc[missing_index, 'WetBulb'] = y_pred.round()

    df_weather.WetBulb = df_weather.WetBulb.astype(float)

    df_weather.PrecipTotal = df_weather.PrecipTotal.str.strip(
        ).str.replace('T', '0.001').str.replace('M', '0.001').astype('float')
    df_weather.Date = pd.to_datetime(df_weather.Date)

    weather_st1 = df_weather[df_weather.Station == 1].drop('Station', axis=1)
    weather_st2 = df_weather[df_weather.Station == 2].drop('Station', axis=1)
    df_weather = weather_st1.merge(
        weather_st2,
        on='Date',
        suffixes=['_1', '_2']
    )

    return df_weather


def preprocess_data(data: dict) -> dict:

    data_preprocessing_pipeline = build_data_preprocessing_pipeline()

    data['train'] = data_preprocessing_pipeline.fit_transform(data['train'])
    data['test'] = data_preprocessing_pipeline.transform(data['test'])

    data['weather'] = clean_weather(data['weather'])

    return data


def split_date(df: pd.DataFrame) -> pd.DataFrame:
    """Splits 'Date' column into seperate columns for
    month, year, week, day of a year

    Args:
        df (pd.DataFrame): data frame
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Week'] = df.Date.dt.isocalendar().week
    df['Dayofyear'] = df['Date'].dt.dayofyear
    return df


class MonthSpeciesTrapTransformer(BaseEstimator, TransformerMixin):
    """Transformer to drop rows with months, species and traps for which
    wnv presence was detected less than 3 times"""

    def __init__(self, columns=['Species', 'Month', 'Trap']):

        self.columns = columns
        self.positive = dict()

    def fit(self, df: pd.DataFrame):

        for col in self.columns:
            virus_detected_cnt = df.groupby(col)['WnvPresent'].sum()
            self.positive[col] = virus_detected_cnt[
                virus_detected_cnt > 2
            ].index.to_list()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in self.columns:
            df = df[df[col].isin(self.positive[col])]

        return df


class SpeciesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.enc = None

    def fit(self, df: pd.DataFrame):
        self.enc = OneHotEncoder(sparse_output=False)
        self.enc.fit(df['Species'].to_frame())

        return self

    def transform(self, df: pd.DataFrame):
        df_species = pd.DataFrame(
            self.enc.transform(df['Species'].to_frame()),
            columns=self.enc.get_feature_names_out()
        )
        return pd.concat(
            [
                df.reset_index(drop=True).drop(['Species'], axis=1),
                df_species
            ],
            axis=1
        )


def remove_address(df: pd.DataFrame) -> pd.DataFrame:
    
    columns_to_drop = ['Address', 'Block', 'Street', 'AddressNumberAndStreet',
                       'AddressAccuracy']

    if 'NumMosquitos' in df.columns:
        columns_to_drop.append('NumMosquitos')

    return df.drop(columns_to_drop, axis=1)
