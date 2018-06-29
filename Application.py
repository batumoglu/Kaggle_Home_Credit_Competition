import Dataset
import ModelPool
from Tasks import TaskData
from MLPipeline import Pipeline

# Initialize ML pipeline
pipeline = Pipeline()

# Define models that are to be included in the ML Pipeline
@pipeline.Model("catboostv1","This model will be trained using CatBoost classifier " +
        " with 5-Fold CV. Predictions will be evaluated based on AUC metric.")
def CatBoost_v1():
    return ModelPool.CatBoost_v1()

@pipeline.Model("lightbgmv1","This model will be trained using LightGBM classifier " +
        " with 5-Fold CV. Predictions will be evaluated based on AUC metric.")
def LisghtGBM_v1():
    return ModelPool.LightGBM_v1()


# Define datasets that are to be included in the ML Pipeline
@pipeline.Dataset("allDatav2","This dataset has been generated from Application dataset" +
        " downloaded from Kaggle HomeCredit competition.")
def AllData_v2():
    return TaskData(Dataset.Load("AllData_v2"), "AllData_v2")

@pipeline.Dataset("allDatav3","This dataset has been generated from Application dataset" +
        " downloaded from Kaggle HomeCredit competition.")
def AllData_v3():
    return TaskData(Dataset.Load("AllData_v3"), "AllData_v3")

@pipeline.Dataset("applicationburo","This dataset has been generated from ApplicationBuro dataset" +
        " downloaded from Kaggle HomeCredit competition.")
def ApplicationBuro():
    return TaskData(Dataset.Load("ApplicationBuro"), "ApplicationBuro")

