import Dataset
import ModelPool
from Tasks import TaskData
from MLPipeline import Pipeline

# Initialize ML pipeline
pipeline = Pipeline()

# Define models that are to be included in the ML Pipeline
@pipeline.Model("catboostv1")
def CatBoost_v1():
    return ModelPool.CatBoost_v1()

@pipeline.Model("lightbgmv1")
def LisghtGBM_v1():
    return ModelPool.LightGBM_v1()


# Define datasets that are to be included in the ML Pipeline
@pipeline.Dataset("allDatav2")
def AllData_v2():
    return TaskData(Dataset.AllData_v2, "AllData_v2")

@pipeline.Dataset("allDatav3")
def AllData_v3():
    return TaskData(Dataset.AllData_v2, "AllData_v3")

@pipeline.Dataset("applicationburo")
def ApplicationBuro():
    return TaskData(Dataset.AllData_v2, "ApplicationBuro")