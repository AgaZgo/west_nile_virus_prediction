from loguru import logger

from src.data import read_raw_data, preprocess_data
from src.features import get_features
from src.training import get_training_data, train_with_hyperparams_tuning
from src.evaluate import evaluate
from src.config import RESAMPLE_METHOD


def run_experiment():
    data = read_raw_data()
    logger.info('Raw data read.')

    data = preprocess_data(data)
    logger.info('Preprocessing finished.')

    df_train, df_test = get_features(data)
    logger.info('Feature engineering finished.')

    features, labels = get_training_data(df_train, method=RESAMPLE_METHOD)
    logger.info('Training data created.')

    model = train_with_hyperparams_tuning(features, labels)
    logger.info('Model trained.')

    feature_names = features.columns.to_list()
    scores = evaluate(model, df_test, feature_names)
    logger.info('Evaluation finished.')
    logger.info(f'Public score: {scores[0]}, private score: {scores[1]}')


if __name__ == "__main__":
    run_experiment()
