from src.data import read_raw_data, preprocess_data
from src.features import get_features
from src.training import get_training_data, train
from src.evaluate import evaluate


data = read_raw_data()

data = preprocess_data(data)

df_train, df_test = get_features(data)

df_train = get_training_data(df_train, method='random_undersample')

model = train(df_train)

scores = evaluate(model, df_train, df_test)

