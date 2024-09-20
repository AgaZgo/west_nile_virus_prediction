from src.data import read_raw_data, preprocess_data
from src.features import get_features
from src.training import get_training_data, train
from src.evaluate import evaluate
from src.config import RESAMPLE_METHOD

data = read_raw_data()

data = preprocess_data(data)

df_train, df_test = get_features(data)

features, labels = get_training_data(df_train, method=RESAMPLE_METHOD)

model = train(features, labels)

feature_names = features.columns.to_list()
scores = evaluate(model, df_test, feature_names)

print(scores)
