import Dataset
from Tasks import TaskScheduler, Session


class Pipeline(object):
    def __init__(self):
        self._models_ = {}
        self._datasets_ = {}
        self._taskscheduler_ = TaskScheduler()
        self._subscriptions_ = {"itemadded":[], "itemremoved":[], "schedulechanged":[]}

    def Model(self, name):
        def ModelDecorator(ModelObject):
            self._models_[name] = ModelObject
        return ModelDecorator

    def Dataset(self, name):
        def DatasetDecorator(DatasetObject):
            self._datasets_[name] = DatasetObject
        return DatasetDecorator

    def Add(self, item):
        if item in self._models_:
            pipeline_item = self._models_[item]
        elif item in self._datasets_:
            pipeline_item = self._datasets_[item]

        try:
            pipeline_item_instance = pipeline_item()
            sch_count = len(self._taskscheduler_.Schedule)
            self._taskscheduler_.Add(pipeline_item_instance)
            self._notify_("itemadded", pipeline_item_instance)
            if len(self._taskscheduler_.Schedule) > sch_count:
                self._notify_("schedulechanged", self._taskscheduler_.Schedule)
        except:
            print("ERROR: An error occured while creating pipeline item instance")
            
    def Remove(self, item):
        try:
            sch_count = len(self._taskscheduler_.Schedule)
            self._taskscheduler_.Remove(item)
            self._notify_("itemremoved", item)
            if len(self._taskscheduler_.Schedule) < sch_count:
                self._notify_("schedulechanged", self._taskscheduler_.Schedule)
        except:
            print("ERROR: An error occured while removing item from pipeline")

    def Subscribe(self, event, handler):
        if event in self._subscriptions_:
            subs = self._subscriptions_[event]
            if not handler in subs:
                subs.append(handler)

    def Unsubscribe(self, event, handler):
        if event in self._subscriptions_:
            subs = self._subscriptions_[event]
            if handler in subs:
                subs.Remove(handler)

    def Run(self):
        session = Session(log=True, log_path="c:\\")
        session.Run(self._taskscheduler_.Schedule)

    def _notify_(self, event, obj):
        for listener in self._subscriptions_[event]:
            listener(obj)

    @property
    def Models(self):
        return self._models_.values()

    @property
    def Datasets(self):
        return self._datasets_.values()