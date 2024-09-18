from src.data import read_raw_data, preprocess_data
from src.features import get_features


data = read_raw_data()

data = preprocess_data(data)

df_train, df_test = get_features(data)

breakpoint()
