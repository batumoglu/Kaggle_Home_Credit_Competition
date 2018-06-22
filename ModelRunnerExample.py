import Dataset
from Tasks import Task
from Tasks import TaskData
from Tasks import Session
from Tasks import TaskScheduler

class KnnModel(Task):
    def Run(self):
        # Specify Task Id
        self.SetId("KnnBasic")

        # Specify parameters used in model
        self.SetParam("n_neighbors", 5)
        self.SetParam("leaf_size", 30)
        self.SetParam("weights", "uniform")

        # Define model here
        x_train = self.Data.X_Train
        x_test = self.Data.X_Test
        y_train = self.Data.Y_Train
        # .........

        # Specify scores
        self.SubmitScore("accuracy",0.76)
        self.SubmitScore("roc_auc",0.68)

dataset = TaskData(Dataset.AllData_v2,"AllData_v2")
task = KnnModel()

ts = TaskScheduler([task],[dataset])
tasks = ts.Compile()
session = Session(log=True,log_path="c:\\")
session.Run(tasks)