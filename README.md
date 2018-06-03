# Home_Credit

## Dataset is imbalanced (8%)

Data Preprocess Options:

1- __ApplicationOnly__: Application train and test are gathered only

2- __ApplicationBuroAndPrev__: Buro and Prev files are added

3- __AllData__: All possible data gathered

## Potential Models:
* LightGBM
* XGBoost
* CatBoost
* StackNet
* Sklearn classification models

## Working on
* Generate more possible types of train data
* Generate more features

## Problems
* Most of the models generate 0.50 probability, on kaggle finetuned models generate differenet probability

## Kaggle Notes
* Most of kernels used one hot encoding for features
* Some tables are consist of time series so one hot encoding is fastest solution
* Result my improve if we find better way

## Methods
1- Generate Pipeline (Gather data, increase models) and develope a data-model matrix
2- ROC of trained models on trained data should be close to ROC of test data
3- kFold may help, fit only may have good results but no guarantee to be good on private label

## Questions
1- In  tree models does it make sense to create dummies or is it better to factorize data

__Links__: 

https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931
