# Home_Credit

## Dataset is imbalanced (8%)

Data Preprocess Options:

1- __ApplicationOnly__: Application train and test are gathered only

2- __ApplicationBuroAndPrev__: Buro and Prev files are added. Other tables contain dummy data

3- __AllData__: All possible data gathered. Other tables contain dummy data

4- __ApplicationBuro__: Buro data added to application. None dummy generated.

5- __ApplicationBuroBalance__: Buro Balance and Buro Added to Application. All value are generated by aggregation. None dummy generated.

6- __AllData_v2__: All tables are used. None dummy generated.



## Potential Models:
* LightGBM
* XGBoost
* CatBoost
* StackNet
* Sklearn classification models

## Working on
* Generate more possible types of train data
* Generate more features

## Kaggle Notes
* Most of kernels used one hot encoding for features
* Some tables are consist of time series so one hot encoding is fastest solution
* Result my improve if we find better way
* Method of winner in previous competition:
https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335

## Methods
1- Generate Pipeline (Gather data, increase models) and develope a data-model matrix
2- ROC of trained models on trained data should be close to ROC of test data
3- kFold may help, fit only may have good results but no guarantee to be good on private label

## Model Template
* Should contain kFolds
* Should have result in train data (AUC on train)
* Optionally have importance matrix (Data/Model matrix will grow so it will not be easy)

## Similar Competitions
* https://www.kaggle.com/c/santander-customer-satisfaction/kernels

## Automated Model Running
Below components have been implemented to reduce the complexity and time consuming process of model training stage, including data collection, fine tunning and result analysis.

1- ModelPool module
   This module serves as a container of all defined models. All models that have been developed to be included in the ML Pipeline should be defined within this module initially.  Model classes must be derived from `Task` object in order for exposing the model ID and the `Run` funcionality to `Session` object and gain access to supplied datasets and result submission functionality. 
   
2- Dataset module
   This module serves as container of all datasets that are to be consumed during training stage. It also allows dataframes to be read/ written from/to disk in a certain file format (.data). Datasets generated on the run (in pandas dataframe format) can be stored on disk for further use in training stage which will also yield to a dramatic reduction in data pre-processing phase.
3- ML Pipeline
   This module provides access to the `Run` functionality to start a training `Session` which runs models as configured. Entire training process and the results are reported to a specified file on disk. 
  
3- Application module 
   ML Pipeline is configured and the training session started on this file.

## Questions
1- In  tree models does it make sense to create dummies or is it better to factorize data

__Links__: 

https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931
