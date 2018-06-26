import Dataset
from Tasks import TaskScheduler, Session


class Pipeline(object):
    def __init__(self):
        self._models_ = {}
        self._datasets_ = {}

    def Model(self, name):
        def ModelDecorator(ModelObject):
            self._models_[name] = ModelObject
            return ModelObject
        return ModelDecorator

    def Dataset(self, name):
        def DatasetDecorator(DatasetObject):
            self._datasets_[name] = DatasetObject
            return DatasetObject
        return DatasetDecorator

    def Run(self, model=None, dataset=None):
        if model is None:
            models = self._models_.values()
        else:
            models = [self._models_[model]]

        if dataset is None:
            datasets = self._datasets_.values()
        else:
            datasets = [self._datasets_[dataset]]

        tasks = TaskScheduler(models, datasets).Compile()
        Session(log=True, log_path="c:\\").Run(tasks)

    @property
    def Models(self):
        return self._models_.keys()

    @property
    def Datasets(self):
        return self._datasets_.keys()