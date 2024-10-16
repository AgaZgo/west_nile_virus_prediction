# West Nile Virus prediction

Machine Learning project for predicting West Nile Virus outbreaks from imbalanced data.

The aim of the project is to build virus presence predictor for mosquitos caught in Chicago and investigate possible benefits of various feature engineering methods on model's performance. 

## Dataset
[Dataset](https://www.kaggle.com/competitions/predict-west-nile-virus/data) used in this project comes from Kaggle competition "West Nile VirusPrediction".

## Metrics and evaluation
Since the data are imbalanced, area under the ROC curve (AUC ROC) is used as evaluation metric.

We evaluate performance of our models on the test set provided by Kaggle. We make submission of our models' predictions through Kaggle API and use 'private score' as an evaluation metric.

For each model we report mean score of 5 runs.

## Feature engineering

1. Basic transformations (performed each time):
    - extracting date components (day of year, week number)
    - one-hot encoding species of moquitos
    - removing trap address 
2. Generated trap features:
    - maximal number of mosquitos caught in a trap in a day
    - probability of detecting virus in a trap
3. Weather features:
    - choosing most relevant weather features based on literature
    - lagged and aggregated weather features based on literature and automated feature selection using Sequential Feature Extraction
4. Number of multirows (leakage feature)


## Modelling
MAchine learning models used and compared:
- logistic regression
- LGBM classifier
                             
We use Optuna for hyperparameter tuning.

## Results
 | model |feature engineering|  private score |
 |--- | ---- | --- | 
 |logistic regression|weather| 0.73 |
 | logistic regression | weather + trap fearures| | ||
 |logistic regression | weather + trap features + number of rows||||
 |logistic regression | weather + trap features + lagged weather||||
|LightGBM | weather + trap features + number of rows + lagged weather||||

## How to run a project
