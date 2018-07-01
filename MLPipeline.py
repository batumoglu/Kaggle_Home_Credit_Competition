import Dataset
from Tasks import TaskScheduler, Session


class Pipeline(object):
    ItemAddedEvent = "itemadded"
    ItemRemovedEvent = "itemremoved"
    ScheduleChangedEvent = "schedulechanged"

    def __init__(self):
        self._models_ = {}
        self._datasets_ = {}
        self._items_ = []
        self._taskscheduler_ = TaskScheduler()
        self._subscriptions_ = {self.ItemAddedEvent:[], self.ItemRemovedEvent:[], self.ScheduleChangedEvent:[]}

    def Model(self, name, description=""):
        def ModelDecorator(ModelObject):
            self._models_[name] = ModelObject
            self._items_.append(PipelineItem("model", name, description))
        return ModelDecorator

    def Dataset(self, name, description=""):
        def DatasetDecorator(DatasetObject):
            self._datasets_[name] = DatasetObject
            self._items_.append(PipelineItem("dataset", name, description))
        return DatasetDecorator

    def Add(self, item):
        import time
        time.sleep(2)
        if item in self._models_:
            pipeline_item = self._models_[item]
        elif item in self._datasets_:
            pipeline_item = self._datasets_[item]

        try:
            pipeline_item_instance = pipeline_item()
            sch_count = len(self._taskscheduler_.Schedule)
            self._taskscheduler_.Add(pipeline_item_instance)
            self._notify_(self.ItemAddedEvent, item)
            if len(self._taskscheduler_.Schedule) > sch_count:
                self._notify_(self.ScheduleChangedEvent, self._taskscheduler_.Schedule)
        except:
            print("ERROR: An error occured while creating pipeline item instance")
            
    def Remove(self, item):
        try:
            sch_count = len(self._taskscheduler_.Schedule)
            self._taskscheduler_.Remove(item)
            self._notify_(self.ItemRemovedEvent, item)
            if len(self._taskscheduler_.Schedule) < sch_count:
                self._notify_(self.ScheduleChangedEvent, self._taskscheduler_.Schedule)
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

    @property
    def Items(self):
        return self._items_


class PipelineItem(object):
    def __init__(self, itemtype, name, description):
        self._type_ = itemtype
        self._name_ = name
        self._description_ = description

    @property
    def Type(self):
        return self._type_

    @property
    def Name(self):
        return self._name_

    @property
    def Description(self):
        return self._description_