import ModelRunner as mr
from ModelRunner import Task
from ModelRunner import TaskData
from ModelRunner import TaskResult
from ModelRunner import Session
from ModelRunner import TaskScheduler

class KnnModel(Task):
    def Run(self, data):
        # Specify Task Id
        self.SetId("KnnBasic")

        # Specify parameters used in model
        self.SetParam("n_neighbors", 5)
        self.SetParam("leaf_size", 30)
        self.SetParam("weights", "uniform")

        # Define model here
        x_train = data.X_Train
        x_test = data.X_Test
        y_train = data.Y_Train
        # .........

        # Specify scores
        self.SubmitScore("accuracy",0.76)
        self.SubmitScore("roc_auc",0.68)

        # Return result
        return TaskResult(self)

x_train = [1,2,3,4,5,6,7]
x_test = [1,2,3]
y_train = [0,0,1,0,0,0,1]

dataset = TaskData(x_train,x_test,y_train)
task = KnnModel()

ts = TaskScheduler([task],[dataset])
tasks = ts.Compile()
session = Session(log=True,log_path="c:\\")
session.Run(tasks)