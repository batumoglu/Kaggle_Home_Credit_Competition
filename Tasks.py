import itertools

class TaskScheduler(object):
    def __init__(self, tasks, datasets):
        self._datasets_ = datasets
        self._tasks_ = tasks
        self._schedule_ = None

    def _ValidateInputs_(self):
        if not isinstance(self._datasets_, list):
            raise ValueError("ArgumentOutOfRange: Input should be of list type")
        if not isinstance(self._tasks_, list):
            raise ValueError("ArgumentOutOfRange: Input should be of list type")
        
        if(len(self._datasets_) * len(self._tasks_) <= 0):
            raise ValueError("ArgumentOutOfRange: Specified arguments should contain at least one element")

        for ds in self._datasets_:
            if not isinstance(ds, TaskData):
                raise ValueError("ArgumentOutOfRange: Invalid dataset type provided. Expected tuple")
        for t in self._tasks_:
            if not isinstance(t, Task):
                raise ValueError("ArgumentOutOfRange: Invalid model type provided. Expected Model type")

    def Compile(self):
        self._ValidateInputs_()
        self._schedule_ = list(itertools.product(self._tasks_, self._datasets_))
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
    def __init__(self):
        self._params_ = dict()
        self._scores_ = dict()
        self._data_ = None

    # This function should not be overrided within subclasses
    def RunTask(self, data):
        self._data_ = data
        self.Run()
        return TaskResult(self)

    # Override this function in subclasses and run your task relevant job here
    def Run(self):
        pass
    
    def SetId(self, id):
        self._Id_ = id

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
    def Id(self):
        return self._Id_

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