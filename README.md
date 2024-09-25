# West Nile Virus prediction

Machine Learning project for predicting West Nile Virus outbreaks from imbalanced data.

The aim of the project is to build virus presence predictor for mosquitos caught in Chicago. 

The focus is on investigating what resampling technique leads to the best results. 

Moreover, we perform extensive feature engineering of weather data to finds out what weather condition and in what time range are the most helpul for predicting probability of virus presence in a given sample.

## Resampling techniques

### I. Undersampling
1. NearMiss
2. Tomek Links
3. Edited Nearest Neighbours
4. One Sided Selection
5. Neighbourhood Cleaning Rule

### II. Oversampling
1. SMOTE
2. ADASYN

### III. Combined methods
1. SMOTE + Tomek Links
2. SMOTE + Edited Nearest Neighbours


## TODO:

- [ ] config to mlflow
- [ ] readme
- [ ] decouple basic eda and advanced eda
- [ ] what is loss function in lgbm?
