# West Nile Virus prediction

Machine Learning project for predicting West Nile Virus outbreaks.

The aim of the project is to build virus presence predictor for mosquitos caught in Chicago and investigate possible benefits of various feature engineering methods on model's performance. 

## Dataset
[Dataset](https://www.kaggle.com/competitions/predict-west-nile-virus/data) used in this project comes from Kaggle competition "West Nile Virus Prediction".

## Metrics and evaluation
Since the data are imbalanced, area under the ROC curve (AUC ROC) is used as evaluation metric.

We evaluate performance of our models on the test set provided by Kaggle. We make submission of our models' predictions through Kaggle API and use 'private score' as an evaluation metric.

For each model we report mean score of 5 runs.

## Feature engineering

1. Basic transformations:
    - extracting date components (e.g. day of year, week number)
    - one-hot encoding of mosquito species
    - removing trap address 
2. Generated trap features:
    - maximal number of mosquitos caught in a trap in a day
    - probability of detecting virus in a trap
3. Weather features:
    - choosing most relevant weather features
    - generating lagged and aggregated weather features
    - feature selection using Sequential Feature Extraction
4. Number of multirows (leakage feature, not available in most real-life use cases)


## Modelling
Machine learning models used and compared:
- logistic regression
- LGBM classifier
                             
We use Optuna for hyperparameter tuning.

## Results


|| model | basic transform.|trap feat.|weather feat.|lagged weather feat.|multirows|AUC_ROC test score (5-run mean)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:--:|
|baseline|logistic regression |+|-|-|-|-| **0.723** |
||logistic regression|+|+|-|-|-| 0.719 |
||logistic regression|+|+|-|-|+| 0.753 |
||logistic regression|+|+|+|-|-| 0.728 |
||logistic regression|+|+|+|+|-| 0.738 |
|best log. regr.|logistic regression|+|+|+|+|+| **0.763** |
|best overall|lightGBM|+|+|+|+|+| **0.773** |



## How to run a project

1. Install [Python Poetry](https://python-poetry.org/)
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
2. cd into project directory and run:
    ```bash
    poetry install
    ```
3. Run mlflow locally:
    ```bash
    make mlflow
    ```
    You can track your experiments at: http://127.0.0.1:8090 
4. Run experiment:
    ```bash
    make run
    ```
You can change configuration of your experiments by alternating values of variables in ```src/config.py```
