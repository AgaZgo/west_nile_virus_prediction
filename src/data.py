from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline

import pandas as pd
import numpy as np

from src.paths import RAW_DATA_DIR

# disable pandas SettingWithCopyWarnings
pd.options.mode.chained_assignment = None


def read_raw_data() -> dict:
    """Reads 3 csv files with data on virus presence and weather conditions
    into a dictionary

    Returns:
        dict: {'train': pd.DataFrame, 'test': pd.DataFrame,
                'weather': pd.DataFrame}
    """

    files = ['train.csv', 'test.csv', 'weather.csv']
    return {
        file_name[:-4]: pd.read_csv(RAW_DATA_DIR / file_name)
        for file_name in files
        }


def build_data_preprocessing_pipeline() -> Pipeline:
    """Builds transformation pipeline for virus presence data.
    Pipeline:
        - splits date into separate columns with date components
        - removes rows with months, species and traps values for which virus
        was detected less than 3 times
        - removes address data

    Returns:
        Pipeline: basic preprocessing pipeline for train and test data
    """

    date_transformer = FunctionTransformer(split_date)
    month_species_trap_filter = MonthSpeciesTrapTransformer()
    address_remover = FunctionTransformer(remove_address)

    return make_pipeline(
        date_transformer,
        month_species_trap_filter,
        address_remover,
        verbose=True
    )


def clean_weather(df_weather: pd.DataFrame) -> pd.DataFrame:
    """Perform weather data cleaning based on
    finding of exploratory data analysis

    Args:
        df_weather (pd.DataFrame): raw weather data

    Returns:
        pd.DataFrame: clean weather data
    """

    # choice of columns to use in the project follows from insights from EDA
    columns_to_stay = ['Station', 'Date', 'Tmax', 'Tmin', 'Tavg', 'DewPoint',
                       'WetBulb', 'PrecipTotal', 'AvgSpeed', 'ResultSpeed',
                       'ResultDir']
    df_weather = df_weather[columns_to_stay]

    # 'Tavg' is an average of 'Tmax' and 'Tmin'.
    # We use it to fill missing values marked with 'M'
    df_weather.Tavg = np.where(
        df_weather.Tavg == 'M',
        df_weather[['Tmax', 'Tmin']].mean(axis=1),
        df_weather.Tavg
    ).astype(np.float64)

    # To fill missing values in 'WetBulb' we will use simple regression model
    # which uses columns 'Tmax', 'Tmin', 'DewPoint' and predicts value of in
    # 'WetBulb'
    lr = LinearRegression()

    train_weather = df_weather[df_weather.WetBulb != 'M'][
        ['Tmax', 'Tmin', 'DewPoint', 'WetBulb']
    ]
    test_weather = df_weather[df_weather.WetBulb == 'M'][
        ['Tmax', 'Tmin', 'DewPoint']
    ]
    y_train = train_weather.pop('WetBulb').astype(np.float64)
    X_train = train_weather
    X_test = test_weather

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    missing_index = df_weather[df_weather.WetBulb == 'M'].index
    df_weather.loc[missing_index, 'WetBulb'] = y_pred.round()

    df_weather['WetBulb'] = df_weather.WetBulb.astype(np.float64)

    X_train = df_weather[df_weather['AvgSpeed'] != 'M'][
        ['ResultSpeed', 'ResultDir', 'AvgSpeed']
        ].astype(np.float64)
    y_train = X_train.pop('AvgSpeed')

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    X_test = df_weather[df_weather['AvgSpeed'] == 'M'][
        ['ResultSpeed', 'ResultDir']]
    missing_value_predictions = lr_model.predict(X_test).round(1)

    m_index = df_weather[df_weather['AvgSpeed'] == 'M'].index
    df_weather.loc[m_index, 'AvgSpeed'] = missing_value_predictions

    df_weather['AvgSpeed'] = df_weather['AvgSpeed'].astype(np.float64)

    # Value 'T' in column 'PrecipTotal' means that there was a trace of
    # precipitation detected. The smallest non zero number in this column
    # is 0.01, thus we use value 0.001 to replace 'T'. We will use the same
    # value to replace 'M' (missing data)
    df_weather['PrecipTotal'] = df_weather.PrecipTotal.str.strip(
        ).str.replace('T', '0.001').str.replace('M', '0.001').astype(
            np.float64)
    df_weather['Date'] = pd.to_datetime(df_weather.Date)

    # merge data from weather station 1 and 2, so it is in one row
    weather_st1 = df_weather[df_weather.Station == 1].drop('Station', axis=1)
    weather_st2 = df_weather[df_weather.Station == 2].drop('Station', axis=1)
    df_weather = weather_st1.merge(
        weather_st2,
        on='Date',
        suffixes=['_1', '_2']
    )

    return df_weather


def preprocess_data(data: dict) -> dict:
    """Performs basic preprocessing of raw data. Only model agnostic
    transformations.

    Args:
        data (dict):  Dictionary of raw data: {'train': pd.DataFrame,
                'test': pd.DataFrame, 'weather': pd.DataFrame}

    Returns:
        dict: Dictionary of preprocessed data: {'train': pd.DataFrame,
            'test': pd.DataFrame, 'weather': pd.DataFrame}
    """

    # build preprocessing pipeline
    data_preprocessing_pipeline = build_data_preprocessing_pipeline()

    # preprocess train and test data
    data['train'] = data_preprocessing_pipeline.fit_transform(data['train'])
    data['test'] = data_preprocessing_pipeline.transform(data['test'])

    # clean weather data
    data['weather'] = clean_weather(data['weather'])

    return data


def split_date(df: pd.DataFrame) -> pd.DataFrame:
    """Splits 'Date' column into seperate columns for
    month, year, week, day of a year
    and adds these components to input dataframe

    Args:
        df (pd.DataFrame): dataframe with additional columns
    """

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Week'] = df.Date.dt.isocalendar().week
    df['Dayofyear'] = df['Date'].dt.dayofyear
    df['Dayofweek'] = df['Date'].dt.dayofweek

    return df


class MonthSpeciesTrapTransformer(BaseEstimator, TransformerMixin):
    """Transformer to drop rows with months, species and traps for which
    wnv presence was detected less than 3 times"""

    def __init__(self, columns=['Species', 'Month']):

        self.columns = columns
        self.values_with_positive_cases = {
            col: [] for col in columns
        }

    def fit(self, df: pd.DataFrame):

        for col in self.columns:

            # get column values with more 2 positive cases
            virus_detected_cnt: pd.Series = df.groupby(col)['WnvPresent'].sum()
            self.values_with_positive_cases[col] = virus_detected_cnt[
                virus_detected_cnt > 0
            ].index.to_list()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in self.columns:
            # leave df rows with values in self.values_with_positive_cases
            df = df[df[col].isin(self.values_with_positive_cases[col])]

        return df


def remove_address(df: pd.DataFrame) -> pd.DataFrame:
    """Removes address columns from dataframe"""

    columns_to_drop = ['Address', 'Block', 'Street', 'AddressNumberAndStreet',
                       'AddressAccuracy']

    return df.drop(columns_to_drop, axis=1)
