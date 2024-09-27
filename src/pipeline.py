from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler


def get_pipeline(clf=None) -> Pipeline:
    steps = [
        ('scaler', MinMaxScaler()),
        ('smote', BorderlineSMOTE(sampling_strategy=0.2, k_neighbors=5)),
        ('tomek', TomekLinks()),
        ('under', RandomUnderSampler(sampling_strategy=0.7)),
    ]
    if clf:
        steps.append(('model', clf))

    pipeline = Pipeline(steps=steps)
    return pipeline
