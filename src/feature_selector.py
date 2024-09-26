from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

from datetime import timedelta
from loguru import logger

import pandas as pd
import numpy as np


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Class to select weather features for modeling"""

    def __init__(
        self,
        weather_df: pd.DataFrame,
        agg_weather_df: pd.DataFrame,
        num_weather_features: int,
        num_agg_features: int,
        selector: str
    ):
        self.weather_df = weather_df
        self.agg_weather_df = agg_weather_df
        self.num_weather_features = num_weather_features
        self.num_agg_features = num_agg_features
        self.selector = selector

        self.selected_weather_cols = []
        self.selected_agg_cols = []

    def fit(
        self,
        df: pd.DataFrame
    ):
        """Selects weather features based of input dataframe

        Args:
            df (pd.DataFrame): dataframe with 'Date' and 'WnvPresent' columns

        """

        # select non-aggregated and non-lagged weather features
        if self.num_weather_features > 0:
            logger.debug(
                'Selecting non-aggregated and non-lagged weather features...'
            )
            self.selected_weather_cols = self.select_weather_features(
                df,
                self.weather_df,
                self.num_weather_features
            )
            logger.debug('Non-aggregated weather features selected')

        # select aggregated and lagged weather features
        if self.num_agg_features > 0:
            logger.debug(
                'Selecting aggregated and lagged weather features...'
            )
            self.selected_agg_cols = self.select_weather_features(
                df,
                self.agg_weather_df,
                self.num_agg_features
            )
            logger.debug('Aggregated and lagged weather features selected')

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns input dataframe merged (on 'Date' column) with selected
        weather features
        """
        df = self.add_max_catch_date(df)

        data_full = pd.merge(
            pd.merge(
                df,
                self.weather_df.reset_index(),
                left_on='MaxCatchDate',
                right_on='Date'
            ),
            self.agg_weather_df.reset_index(),
            left_on='MaxCatchDate',
            right_on='Date'
        )
        df = data_full[
            [
                *df.columns.to_list(),
                *self.selected_weather_cols,
                *self.selected_agg_cols
            ]
        ]
        df.drop('MaxCatchDate', axis=1, inplace=True)
        return df

    def select_weather_features(
        self,
        df: pd.DataFrame,
        weather_df: pd.DataFrame,
        num_features: int
    ) -> list:
        """Selects best weather features according to specified selector model

        Args:
            df (pd.DataFrame): Dataframe with virus presence data
            weather_df (pd.DataFrame): Weather data from which features will
                                       will be selected. Date as index.
            num_features (int): Number of features to select

        Returns:
            list: List of selected weather features
        """
        df = pd.merge(df, weather_df.reset_index(), on='Date')
        X_train = df[weather_df.columns]
        y_train = df['WnvPresent']

        if self.selector == 'xgb':
            classifier = XGBClassifier()

        logger.debug(f'Selecting {num_features} features with RFE...')

        selector = RFE(
            classifier,
            n_features_to_select=num_features
        ).fit(X_train, y_train)

        support = selector.support_
        selected_features = weather_df.columns[support].to_list()
        logger.debug(f'Selected weather features: {selected_features}')
        return selected_features

    def add_max_catch_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 'MaxCatchDate' columns based on 'Date'. For Mondays, Tuesdays
        and Wednesdays 'MaxCatchDate'='Date'. For Thursdays and Fridays
        'MaxCatchDate' is the date of Wednesday the same week.

        Args:
            df (pd.DataFrame): Dataframe containing columns 'Date' and
                               'Dayofweek'

        Returns:
            pd.DataFrame: Dataframe with 'MaxCatchDate' added
        """

        df['MaxCatchDate'] = np.where(
            df.Dayofweek < 3,
            df.Date,
            np.where(
                df.Dayofweek == 3,
                df.Date - timedelta(days=1),
                df.Date - timedelta(days=2)
            )
        )

        return df
