import pandas as pd
import subprocess

from src.paths import RAW_DATA_DIR, SUBMISSION_DIR


def evaluate(model, df_test, feature_names):
    df_test = df_test[feature_names + ['Id']]
    predictions = model.predict_proba(df_test.drop(['Id'], axis=1))[:, 1]
    df_test['proba'] = predictions

    sample = pd.read_csv(RAW_DATA_DIR / 'sampleSubmission.csv')
    predictions = df_test[['Id', 'proba']]

    results = pd.merge(sample, predictions, on='Id', how='outer').fillna(0)
    results['WnvPresent'] = results['proba']
    results.drop('proba', axis=1, inplace=True)

    results.to_csv(SUBMISSION_DIR / 'test_results.csv', index=False)

    subprocess.run(
        [
            "kaggle", "competitions", "submit", "-c",
            "predict-west-nile-virus", "-f",
            "/home/aga/repos/west_nile_virus/data/submission/test_results.csv",
            "-m", "test"
        ],
        shell=True
    )
    s = subprocess.check_output(
            "kaggle competitions submissions -c predict-west-nile-virus",
            text=True,
            shell=True
        )

    public_score = s.split('\n')[3].split()[-2]
    private_score = s.split('\n')[3].split()[-1]
    print(f"Public score: {public_score}\nPrivate score: {private_score}")
