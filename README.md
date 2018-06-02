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

## Questions
1- In  tree models does it make sense to create dummies or is it better to factorize data

__Links__: 

https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931
