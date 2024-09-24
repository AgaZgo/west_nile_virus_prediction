from typing import Tuple
from time import sleep
from loguru import logger

import pandas as pd
import subprocess

from src.paths import RAW_DATA_DIR, SUBMISSION_DIR


def evaluate(
    model,
    df_test: pd.DataFrame,
    feature_names: list
) -> Tuple[float]:
    """Generates predictions and submits them to Kaggle via Kaggle API.
    Kaggle credetials in ~/.kaggle/kaggle.json file are required

    Args:
        model: model compatible with sklearn API
        df_test (pd.DataFrame): Test data
        feature_names (list): List of features used during training

    Returns:
        Tuple[float]: Scores from Kaggle: (public_score, private_score)
                    Public score: ROC_AUC on smaller subset of test data
                    Private score: ROC_AUC on larger subset of test data
    """
    # get predictions
    df_test = df_test[feature_names + ['Id']]
    predictions = model.predict_proba(df_test.drop(['Id'], axis=1))[:, 1]
    df_test['proba'] = predictions
    predictions = df_test[['Id', 'proba']]

    # read sampleSubmission file for correct submission format
    sample = pd.read_csv(RAW_DATA_DIR / 'sampleSubmission.csv')

    # fill in results dataframe with predictions
    # where there is no prediction fill with 0
    results = pd.merge(sample, predictions, on='Id', how='outer').fillna(0)
    results['WnvPresent'] = results['proba']
    results.drop('proba', axis=1, inplace=True)

    # save submission file
    results.to_csv(SUBMISSION_DIR / 'test_results.csv', index=False)

    # send submission file to Kaggle
    subprocess.run(
        "kaggle competitions submit -c predict-west-nile-virus -f \
            /home/aga/repos/west_nile_virus/data/submission/test_results.csv \
                -m test",
        shell=True,
        executable="/bin/bash"
    )
    logger.debug('File pushed to Kaggle API')

    sleep(1)

    # check Kaggle scores
    s = subprocess.check_output(
            "kaggle competitions submissions -c predict-west-nile-virus",
            text=True,
            shell=True
        )

    public_score = s.split('\n')[3].split()[-2]
    private_score = s.split('\n')[3].split()[-1]

    return public_score, private_score
