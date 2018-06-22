import Dataset
from ModelPool import CatBoost_v1, LightGBM_v1
from Tasks import TaskData, TaskResult, TaskScheduler, Session


allData = TaskData(Dataset.AllData, "AllData")
allData_v2 = TaskData(Dataset.AllData_v2, "AllData_v2")
allData_v3 = TaskData(Dataset.AllData_v3, "AllData_v3")

datasets = [allData, allData_v2, allData_v3]
models = [CatBoost_v1(), LightGBM_v1()]

tasks = TaskScheduler(models, datasets).Compile()
Session(log=True,log_path="c:\\").Run(tasks)


class Pipeline(object):
    def __init__(self):
        self._models_ = []
        self._datasets_ = []

    def Model(self, Task):
        def ModelDecorator(ModelObject):
            def ModelWrapper():
                self._models_.append(ModelObject())
            return ModelWrapper
        return ModelDecorator
    
    def Dataset(self, TaskData):
        def DatasetDecorator(DatasetObject):
            def DatasetWrapper():
                self._datasets_.append(DatasetObject())
            return DatasetWrapper
        return DatasetDecorator

    def Run(self):
        tasks = TaskScheduler(self._models_, self._datasets_).Compile()
        Session(log=True, log_path="c:\\").Run(tasks)