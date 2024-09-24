from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SequentialFeatureSelector
from xgboost import XGBClassifier
from loguru import logger

import pandas as pd


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

        data_full = pd.merge(
            pd.merge(df, self.weather_df.reset_index(), on='Date'),
            self.agg_weather_df.reset_index(),
            on='Date'
        )
        df = data_full[
            [
                *df.columns.to_list(),
                *self.selected_weather_cols,
                *self.selected_agg_cols
            ]
        ]

        return df

    def select_weather_features(
        self,
        df: pd.DataFrame,
        weather_df: pd.DataFrame,
        num_features: int
    ) -> list:
        """

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

        logger.debug(f'Sequentially selecting {num_features} features..')
        sfs_forward = SequentialFeatureSelector(
            classifier,
            n_features_to_select=num_features,
            direction='forward',
            n_jobs=-1
        ).fit(X_train, y_train)

        return weather_df.columns[sfs_forward.get_support()].to_list()
