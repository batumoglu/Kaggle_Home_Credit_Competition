import ModelRunner as mr
from ModelRunner import Task
from ModelRunner import Session
from ModelRunner import TaskScheduler

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

x_train = [1,2,3,4,5,6,7]
x_test = [1,2,3]
y_train = [0,0,1,0,0,0,1]

dataset = (x_train,x_test,y_train)
task = KnnModel()

ts = TaskScheduler([task],[dataset])
tasks = ts.Compile()
session = Session(log=True,log_path="c:\\")
session.Run(tasks)