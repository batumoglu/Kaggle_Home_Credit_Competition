import itertools
import json
from datetime import datetime
from time import time

class TaskScheduler(object):
    def __init__(self):
        self._datasets_ = {}
        self._tasks_ = {}
        self._schedule_ = []

    def Add(self, item):
        if isinstance(item,Task) and item.Name not in self._tasks_:
            self._tasks_[item.Name] = item
        elif isinstance(item, TaskData) and item.Name not in self._datasets_:
            self._datasets_[item.Name] = item
        self._compile_()

    def Remove(self, item):
        if isinstance(item,Task) and item.Name in self._tasks_:
            del self._tasks_[item.Name]
        elif isinstance(item, TaskData) and item.Name in self._datasets_:
            del self._datasets_[item.Name]
        elif isinstance(item,str):
            if item in self._tasks_:
                del self._tasks_[item]
            elif item in self._datasets_:
                del self._datasets_[item]
        self._compile_()

    def _compile_(self):
        if self._schedule_ is not None:
            del self._schedule_

        self._schedule_ = list(itertools.product(self._tasks_.values(), self._datasets_.values()))
        return self._schedule_

    @property
    def Schedule(self):
        return self._schedule_

class Session(object):
    def __init__(self, log=False, log_path=""):
        self._log_ = log
        self._log_path_ = log_path

    def Run(self, schedule):
        print("\r\n")
        print("                SESSION START                ")
        print("*********************************************")
        for task, data in schedule:
            task_result = task.RunTask(data)
            if self._log_:
                self._LogResult_(task_result)

    def _LogResult_(self, task_result):
        print("ID\t: " + task_result.Id)
        print("Params\t: " + str(task_result.Parameters))
        print("Dataset\t: " + task_result.Dataset)
        print("Scores\t: " + str(task_result.Scores))
        print("Task ran for 50 seconds on 08-06-2018 16:21")
        print("---------------------------------------------")

class Task(object):
    def __init__(self, name):
        self._params_ = dict()
        self._scores_ = dict()
        self._data_ = None
        self._name_ = name

    # This function should not be overrided within subclasses
    def RunTask(self, data):
        self._data_ = data
        self.Run()
        return TaskResult(self)

    # Override this function in subclasses and run your task relevant job here
    def Run(self):
        pass
    
    def SetName(self, name):
        self._name_ = name

    def SetParam(self, name, value):
        self._params_[name] = value

    def SubmitScore(self, metric, score):
        self._scores_[metric] = score

    def SetDescription(self, description):
        self.__description__ = description

    @property
    def Data(self):
        return self._data_

    @property
    def Name(self):
        return self._name_

    @property
    def Parameters(self):
        return self._params_

    @property
    def Scores(self):
        return self._scores_

    @property
    def Description(self):
        return self.__description__

class TaskData(object):
    def __init__(self, data, name):
        self._xtrain_, self._xtest_, self._ytrain_ = data
        self._name_ = name

    @property
    def X_Train(self):
        return self._xtrain_

    @property
    def X_Test(self):
        return self._xtest_

    @property
    def Y_Train(self):
        return self._ytrain_

    @property
    def Name(self):
        return self._name_

class TaskResult(object):
    def __init__(self, task):
        self._task_ = task
    
    @property
    def Id(self):
        return self._task_.TaskId
 
    @property
    def Parameters(self):
        return self._task_.Parameters

    @property
    def Dataset(self):
        return self._task_.Data.Name
    
    @property
    def Scores(self):
        return self._task_.Scores
    
 class Result(object):
    def __init__(self, metrics):
        self._metrics_ = metrics
        self._results_ = {}

    def Submit(self, metrics, params, scores):
        timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        self._results_[timestamp] = {}
        for metric in self._metrics_:
            metric_idx = self._getindex_(metrics, metric)
            if metric_idx >= 0:
                self._results_[timestamp][metric] = {"params":params[metric_idx], "score":scores[metric_idx]}
            else:
                self._results_[timestamp][metric] = {"params":{}, "score":0}

    def ToJson(self):
        return json.dumps(self.Submissions)

    def _getindex_(self, lst, val):
        try:
            return lst.index(val)
        except:
            return -1

    @property
    def Submissions(self):
        scores = {}
        for score in self._results_.values():
            for metric in score.keys():
                if metric in scores:
                    scores[metric].append(score[metric])
                else:
                    scores[metric] = [score[metric]]

        return {"timestamps":list(self._results_.keys()), "scores":scores}
