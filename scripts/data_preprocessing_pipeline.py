from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline

import pandas as pd

from src.paths import RAW_DATA_DIR, PREPROCESSED_DATA_DIR
from src.data import RowFilterTransformer, SpeciesEncoder
from src.data import split_date, select_columns


data_train = pd.read_csv(RAW_DATA_DIR / 'train.csv')
data_test = pd.read_csv(RAW_DATA_DIR / 'test.csv')


def build_data_preprocessing_pipeline() -> Pipeline:

    date_transformer = FunctionTransformer(split_date)
    row_filter_transformer = RowFilterTransformer()
    species_encoder = SpeciesEncoder()
    cols_selector = FunctionTransformer(select_columns)

    return make_pipeline(
        date_transformer,
        row_filter_transformer,
        species_encoder,
        cols_selector,
        memory='cache',
        verbose=True
    )


data_preprocessing_pipeline = build_data_preprocessing_pipeline()

data_train = data_preprocessing_pipeline.fit_transform(data_train)
data_test = data_preprocessing_pipeline.transform(data_test)

data_train.to_pickle(PREPROCESSED_DATA_DIR / 'preprocessed_train.pkl')
data_test.to_pickle(PREPROCESSED_DATA_DIR / 'preprocessed_test.pkl')
